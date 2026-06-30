//! Shared spec for the `slack_raw` append-log collection.
//!
//! Real-time Slack messages are written as raw, single, **UNINDEXED** records
//! to `slack_raw` (an append log). Batching, rendering, and generation are
//! deferred to a downstream attached function rather than happening at ingest
//! time. Foundation `/init` creates this collection record-only and wires it
//! as an attached-function input.
//!
//! This lives in `chroma-types` (the low-level crate both Foundation and
//! hosted-chroma's sync service already depend on) so the collection name and
//! schema are a single source of truth shared across repos — the sync ingest
//! endpoint can create `slack_raw` identically to `/init` without depending on
//! the heavier `foundation-api` crate. Per-record metadata is the producer's
//! responsibility (written at ingest time by sync), not part of this spec.

use crate::Schema;

/// Name of the raw Slack append-log collection.
pub const SLACK_RAW_COLLECTION_NAME: &str = "slack_raw";

/// Schema for the `slack_raw` collection: record-only, with every index
/// disabled (inverted, FTS, dense vector, and sparse vector). Records are
/// stored verbatim and never indexed at ingest; all batching, rendering, and
/// embedding happen downstream in the attached function.
pub fn slack_raw_schema() -> Schema {
    Schema::new_record_only()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slack_raw_collection_name_is_slack_raw() {
        assert_eq!(SLACK_RAW_COLLECTION_NAME, "slack_raw");
    }

    #[test]
    fn slack_raw_schema_disables_all_indexes() {
        let schema = slack_raw_schema();

        // No active sparse or FTS indexes.
        assert!(
            !schema.is_sparse_index_enabled(),
            "slack_raw must not enable a sparse vector index"
        );
        assert!(
            !schema.is_fts_enabled(),
            "slack_raw must not enable full-text search"
        );

        // Dense vector index disabled in the defaults.
        let float_list = schema
            .defaults
            .float_list
            .as_ref()
            .expect("schema defaults must carry a dense vector index");
        assert!(
            !float_list.vector_index.as_ref().unwrap().enabled,
            "slack_raw must not enable a dense vector index"
        );

        // Metadata inverted index disabled in the defaults.
        let string = schema
            .defaults
            .string
            .as_ref()
            .expect("schema defaults must carry a string value type");
        assert!(
            !string.string_inverted_index.as_ref().unwrap().enabled,
            "slack_raw must not enable a string inverted index"
        );
    }
}
