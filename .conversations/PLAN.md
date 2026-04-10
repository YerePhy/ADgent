# Neuroimaging Analysis Pivot — Roadmap

## Current State

The project is a LangGraph ReAct agent with Gradio frontend, specialized for Alzheimer's/Tau PET research. Key facts relevant to this pivot:

- **nilearn + nibabel already declared** as optional deps in `pyproject.toml` but unused
- **No file upload UI** — Gradio makes this trivial to add
- **Tools are factory-pattern extensible** — adding new tools is clean
- **Single Docker container** — needs rethinking for neuroimaging compute
- **No S3, no EC2 orchestration, no job queue** — all to be built

---

## Architecture Decisions

Two critical design choices drive everything else:

### 1. Where do neuroimaging tools run?

| Option | Pros | Cons |
|---|---|---|
| **Same container** | Simple | Too heavy, blocks agent, not scalable |
| **Sidecar containers (docker socket)** | Low latency, no cloud cost | Bad isolation, agent host needs beefy hardware |
| **On-demand EC2 + Docker** | Scales, GPU-capable, cost-controlled | Complex orchestration, cold start latency |
| **AWS Batch** | Managed queuing + retries + scaling | Less control, slight added complexity |

**Recommendation: AWS Batch** over raw EC2. It handles the instance lifecycle, retry logic, job queuing, and Docker image pulling. You define a Job Definition (FastSurfer image), submit a job, and poll. Far less boilerplate than managing EC2 directly with boto3. Raw EC2 is only better if you need persistent state across jobs (e.g., resumable FreeSurfer runs).

> **Warning:** FastSurfer needs GPU (g4dn.xlarge ~$0.50/hr). SPM standalone uses MATLAB Compiler Runtime — no license needed but the MCR image is ~10 GB. These cold starts can be 5-10 min. Factor that into UX.

### 2. How does the agent interact with long-running jobs?

FastSurfer can take 1+ hours. The ReAct loop cannot block for that long. An **async job pattern** is required:

```
Agent submits job → gets job_id → returns to user
User asks "is it done?" → agent polls status → returns result when ready
```

The agent needs: a `submit_job` tool, a `get_job_status` tool, and a `get_job_results` tool. Job state lives in DynamoDB or a new SQLite table.

---

## Roadmap

### Phase 1 — File Upload + NIfTI Inspection
*Prerequisite for everything. No cloud costs. Delivers immediate value.*

1. Add `gr.File` to Gradio with `.nii/.nii.gz` filter
2. On upload: store locally (or S3 for prod), associate file path with thread ID in SQLite (new `uploads` table)
3. Create `inspect_nifti` tool — nibabel reads header (dims, voxel size, TR, affine, data type)
4. Create `visualize_nifti` tool — nilearn generates orthogonal slice PNG, returns it to Gradio as `gr.Image`
5. Add a results panel (lateral tab in Gradio) for images/tables
6. Update the system prompt — neuroimaging-focused, aware of uploaded file context

**Technology:** nibabel, nilearn, S3 via boto3 (or local path for dev), SQLite for file metadata.

---

### Phase 2 — Python Sandbox
*Enables the agent to run custom analysis without pre-defined tools.*

Options:
- **E2B Code Interpreter** — managed, excellent DX, ~$0.10/session, handles deps, has filesystem
- **AWS Lambda** — 15-min limit, 10 GB memory, no GPU, cold starts
- **Subprocess + seccomp/cgroups** — DIY, fragile, security risk if not careful

**Recommendation: E2B** for speed of iteration. If cost or vendor lock-in is a concern, containerized subprocess with resource limits is the fallback. The key guardrails are: no network access, no shell escape, CPU/memory caps, file access scoped to the user's upload directory.

---

### Phase 3 — AWS Batch + Neuroimaging Tools
*Heavy compute. Requires explicit user consent before launching.*

1. Design IAM roles (agent gets `batch:SubmitJob`, `s3:PutObject` — principle of least privilege)
2. Create Job Definitions for each tool (FastSurfer, FSL, ANTs; SPM standalone)
3. Build `launch_neuroimaging_job` tool — **always asks user confirmation before submitting** (tool surfaces cost estimate + instance type to user)
4. Build `check_job_status` and `get_results` tools
5. Results written to S3, presigned URL returned to user or loaded into results panel
6. Job record in DynamoDB: job_id, thread_id, tool, status, timestamps, cost estimate

> **Warning:** SPM25 may not be production-ready yet — SPM12 standalone with MCR is the safe choice. Verify Docker image availability before committing to it.

---

### Phase 4 — Monitoring + Error Handling
1. CloudWatch alarms on stuck jobs (timeout → auto-terminate + notify agent)
2. Budget alerts via AWS Budgets (hard cap on EC2/Batch spend per month)
3. LangGraph error nodes — graceful degradation when tools fail
4. Job retry logic with exponential backoff (Batch handles this natively)

---

## First Step

**Phase 1, Steps 1–2 in parallel:**

- Wire up `gr.File` → local storage → nibabel `inspect_nifti` tool
- Add a results panel to the Gradio UI

This is self-contained, zero cloud cost, requires no new infrastructure, and validates the full stack (upload → agent → tool → visualization) before touching AWS.
