"""Unit tests for backend.uploads — SqliteUploadStore and factory."""

import sqlite3

import pytest

from backend.uploads import SqliteUploadStore, UploadRecord, create_upload_store


@pytest.fixture()
def store() -> SqliteUploadStore:
    conn = sqlite3.connect(":memory:")
    return SqliteUploadStore(conn)


def test_save_upload_returns_record(store: SqliteUploadStore):
    record = store.save_upload("thread_1", "key/file.nii", "scan.nii")

    assert isinstance(record, UploadRecord)
    assert record.thread_id == "thread_1"
    assert record.store_key == "key/file.nii"
    assert record.filename == "scan.nii"
    assert record.id is not None
    assert record.uploaded_at != ""


def test_list_by_thread_empty(store: SqliteUploadStore):
    assert store.list_by_thread("thread_x") == []


def test_list_by_thread_returns_saved_records(store: SqliteUploadStore):
    store.save_upload("thread_1", "key1", "a.nii")
    store.save_upload("thread_1", "key2", "b.nii")

    records = store.list_by_thread("thread_1")

    assert len(records) == 2
    assert {r.filename for r in records} == {"a.nii", "b.nii"}


def test_list_by_thread_isolates_threads(store: SqliteUploadStore):
    store.save_upload("thread_A", "key_a", "a.nii")
    store.save_upload("thread_B", "key_b", "b.nii")

    assert len(store.list_by_thread("thread_A")) == 1
    assert len(store.list_by_thread("thread_B")) == 1
    assert store.list_by_thread("thread_A")[0].filename == "a.nii"


def test_save_upload_duplicate_store_key_raises(store: SqliteUploadStore):
    store.save_upload("thread_1", "same_key", "first.nii")

    with pytest.raises(sqlite3.IntegrityError):
        store.save_upload("thread_1", "same_key", "second.nii")


def test_save_upload_assigns_incrementing_ids(store: SqliteUploadStore):
    r1 = store.save_upload("t", "k1", "a.nii")
    r2 = store.save_upload("t", "k2", "b.nii")

    assert r2.id > r1.id


def test_create_upload_store_sqlite_backend():
    conn = sqlite3.connect(":memory:")
    store = create_upload_store("sqlite", conn=conn)
    assert isinstance(store, SqliteUploadStore)


def test_create_upload_store_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown upload store backend"):
        create_upload_store("postgres")
