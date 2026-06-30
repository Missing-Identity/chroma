use super::init_schema::{
    foundation_collection_schema, qwen_embedding_function, splade_embedding_function,
    CollectionEmbeddingFunctions,
};
use super::whoami::whoami_and_authorize;
use crate::{
    auth::AuthzAction, config::FoundationConfig, errors::ServerError, server::FoundationApiServer,
};
use axum::{extract::State, http::HeaderMap, Json};
use chroma_error::{ChromaError, ErrorCodes};
use chroma_sysdb::SysDb;
use chroma_types::{
    slack_raw_schema, Collection, CollectionUuid, CreateDatabaseError, DatabaseName, KnnIndex,
    Metadata, MetadataValue, Schema, CHROMA_GROUP_CHUNK_SIBLINGS_KEY, SLACK_RAW_COLLECTION_NAME,
};
use frontend_core::{
    attached_function_ops,
    collection_ops::{
        plan_create_collection, supported_segment_types, ExecutorKind, TenantFeatureFlags,
    },
    foundation::source_kind_for_collection_name,
    retry::retry_transient,
};
use serde::Serialize;
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Serialize)]
pub struct FoundationInitResponse {
    pub tenant: String,
    pub user_id: String,
    pub database: String,
    pub database_id: String,
    pub wiki_collection_id: String,
    pub wiki_revisions_collection_id: String,
    pub currents_collection_id: String,
    pub file_uploads_collection_id: String,
    pub agent_sessions_collection_id: String,
    /// Id of the UNINDEXED `slack_raw` append-log collection. Created
    /// record-only and wired as an attached-function input in place of the
    /// old indexed `slack` source.
    pub slack_raw_collection_id: String,
    /// Source collection name -> id for each ensured source collection
    /// (notion, gdrive, …). Each carries the chunk-sibling grouping flag.
    pub source_collection_ids: std::collections::HashMap<String, String>,
}

/// `POST /api/init` — idempotent bootstrap for a team's Foundation
/// workspace. Ensures the configured Foundation database and the wiki +
/// wiki_revisions collections (names overridable via
/// `CHROMA_FOUNDATION__*` env vars) exist in the tenant resolved from the
/// auth context. Safe to call repeatedly.
#[tracing::instrument(name = "foundation_init", skip_all, err(Display))]
pub async fn foundation_init(
    headers: HeaderMap,
    State(server): State<FoundationApiServer>,
) -> Result<Json<FoundationInitResponse>, ServerError> {
    let identity =
        whoami_and_authorize(&*server.auth, &headers, AuthzAction::InitFoundation).await?;
    let tenant = identity.tenant;
    let user_id = identity.user_id;
    tracing::info!(
        tenant = %tenant,
        user_id = %user_id,
        database = %server.config.foundation.database_name,
        "foundation init starting"
    );

    let _guard =
        server.scorecard_request(&["op:foundation_init", &format!("tenant:{}", tenant)])?;

    let foundation_cfg = &server.config.foundation;
    let db_name = DatabaseName::new(&foundation_cfg.database_name)
        .ok_or(FoundationInitError::DatabaseNameTooShort)?;

    let mut sysdb = server.sysdb.clone();
    let database_id = ensure_database(&mut sysdb, db_name.clone(), tenant.clone()).await?;

    // Wiki collections are the attached function's *output*; they don't
    // need chunk-sibling grouping (no end-of-job marker is read from them).
    let wiki = ensure_collection(
        &mut sysdb,
        tenant.clone(),
        db_name.clone(),
        &foundation_cfg.wiki_collection,
        None,
        // NOTE(hammadb): Foundation uses Qwen0.6B by default which is 1024 dims
        Some(1024),
        CollectionEmbeddingFunctions {
            dense: Some(qwen_embedding_function()),
            sparse: Some(splade_embedding_function()),
        },
    )
    .await?;
    let wiki_revisions = ensure_collection(
        &mut sysdb,
        tenant.clone(),
        db_name.clone(),
        &foundation_cfg.wiki_revisions_collection,
        None,
        Some(1),
        CollectionEmbeddingFunctions::default(),
    )
    .await?;
    // Currents records carry their payload in metadata and are only ever
    // fetched by metadata (never vector-searched), so the collection has no
    // embedding function. Pin the dense index to a single dimension for the
    // derived records written by the currents function.
    let currents =
        ensure_currents_collection(&mut sysdb, tenant.clone(), db_name.clone(), foundation_cfg)
            .await?;

    // Attach revision_history to the wiki collection so every mutation is
    // archived into wiki_revisions automatically on compaction.
    ensure_revision_history_function(
        &mut sysdb,
        tenant.clone(),
        wiki.collection_id,
        foundation_cfg,
    )
    .await?;
    ensure_currents_function(
        &mut sysdb,
        tenant.clone(),
        wiki.collection_id,
        foundation_cfg,
    )
    .await?;

    // Private (per-user) collections — namespaced by user_id so each team
    // member gets their own isolated collection for uploads and traces.
    let file_uploads_name = format!("{}_{}", foundation_cfg.file_uploads_collection, user_id);
    let file_uploads = ensure_collection(
        &mut sysdb,
        tenant.clone(),
        db_name.clone(),
        &file_uploads_name,
        None,
        Some(1024),
        CollectionEmbeddingFunctions {
            dense: Some(qwen_embedding_function()),
            sparse: Some(splade_embedding_function()),
        },
    )
    .await?;

    // The agent_sessions collection is wired into the sources->wiki function
    // below, so it carries the chunk-sibling grouping flag like the other
    // source collections (keeps a job's chunk records in one partition and
    // surfaces the trailing end-of-job marker after every sibling chunk).
    let agent_sessions_name = format!("{}_{}", foundation_cfg.agent_sessions_collection, user_id);
    let agent_sessions = ensure_collection(
        &mut sysdb,
        tenant.clone(),
        db_name.clone(),
        &agent_sessions_name,
        Some(group_chunk_siblings_metadata()),
        Some(1024),
        CollectionEmbeddingFunctions {
            dense: Some(qwen_embedding_function()),
            sparse: Some(splade_embedding_function()),
        },
    )
    .await?;

    // Indexed source collections (notion, gdrive, …) are *extra* inputs to the
    // attached function. They carry the chunk-sibling grouping flag so a job's
    // chunk records stay in one partition and the trailing end-of-job marker on
    // `{base}-0` is observed after every sibling chunk (ADR 0001 §6). All
    // inputs share one async attached function; extras are added via
    // `add_input()` below.
    let mut source_collection_ids = HashMap::new();
    let mut source_collections = Vec::new();
    for source_name in &foundation_cfg.source_collections {
        let source = ensure_collection(
            &mut sysdb,
            tenant.clone(),
            db_name.clone(),
            source_name,
            Some(group_chunk_siblings_metadata()),
            source_dimension(source_name),
            CollectionEmbeddingFunctions::default(),
        )
        .await?;
        source_collections.push((source_name.clone(), source.collection_id));
        source_collection_ids.insert(source_name.clone(), source.collection_id.to_string());
    }

    // Real-time Slack messages land in `slack_raw` as raw, single, UNINDEXED
    // records (an append log) — batching/rendering/generation are deferred to
    // the attached function downstream. Unlike the indexed sources it is
    // created record-only (no embedding function, no dimension). It also does
    // NOT carry the chunk-sibling grouping flag: each message is its own
    // single record, so there are no sibling chunks to keep in one partition.
    let slack_raw = ensure_record_only_collection(
        &mut sysdb,
        tenant.clone(),
        db_name.clone(),
        SLACK_RAW_COLLECTION_NAME,
        None,
    )
    .await?;

    // `slack_raw` is the attached function's *base* input, in place of the old
    // indexed `slack` source. It is the right base for two reasons:
    //  - it is always created (above), so the function is always created and
    //    every input is wired even when `source_collections` is empty; and
    //  - it is a fixed collection, so the base — which keys the function's
    //    identity in sysdb — is stable across `source_collections` changes,
    //    keeping repeated `/init` calls idempotent.
    // Its source_kind resolves to `slack`, so the generation contract is
    // unchanged from the old `slack` base.
    ensure_attached_function(
        &mut sysdb,
        tenant.clone(),
        slack_raw.collection_id,
        SLACK_RAW_COLLECTION_NAME,
        foundation_cfg,
    )
    .await?;

    // Add the indexed sources and the per-user coding-agent traces collection
    // as extra inputs to the same sources->wiki function, so they flow into the
    // shared wiki output alongside slack_raw.
    for (_, source_collection_id) in &source_collections {
        attached_function_ops::add_attached_function_input(
            &mut sysdb,
            foundation_attached_function_name(),
            slack_raw.collection_id,
            *source_collection_id,
            db_name.clone(),
        )
        .await?;
    }
    attached_function_ops::add_attached_function_input(
        &mut sysdb,
        foundation_attached_function_name(),
        slack_raw.collection_id,
        agent_sessions.collection_id,
        db_name.clone(),
    )
    .await?;

    tracing::info!(
        tenant = %tenant,
        num_source_collections = source_collection_ids.len(),
        "foundation init complete"
    );

    Ok(Json(FoundationInitResponse {
        tenant,
        user_id,
        database: foundation_cfg.database_name.clone(),
        database_id: database_id.to_string(),
        wiki_collection_id: wiki.collection_id.to_string(),
        wiki_revisions_collection_id: wiki_revisions.collection_id.to_string(),
        currents_collection_id: currents.collection_id.to_string(),
        file_uploads_collection_id: file_uploads.collection_id.to_string(),
        agent_sessions_collection_id: agent_sessions.collection_id.to_string(),
        slack_raw_collection_id: slack_raw.collection_id.to_string(),
        source_collection_ids,
    }))
}

/// Dense-index dimensionality to pin a source collection to.
///
/// Most sources (e.g. notion) carry 1024-dim vectors supplied by the
/// writer. The Google Drive source instead carries no vectors of its own —
/// the caller upserts records without embeddings — so it is pinned to a
/// single dimension (with no embedding function, like currents /
/// wiki_revisions) to keep the dense index trivially small.
fn source_dimension(source_name: &str) -> Option<i32> {
    match source_kind_for_collection_name(source_name) {
        Ok("google_drive") => Some(1),
        _ => Some(1024),
    }
}

/// Collection metadata that opts a source collection into chunk-sibling
/// grouping during compaction/partitioning (see
/// [`chroma_types::CHROMA_GROUP_CHUNK_SIBLINGS_KEY`]).
fn group_chunk_siblings_metadata() -> Metadata {
    let mut metadata = HashMap::new();
    metadata.insert(
        CHROMA_GROUP_CHUNK_SIBLINGS_KEY.to_string(),
        MetadataValue::Bool(true),
    );
    metadata
}

/// Raised when `/init` needs the attached-function endpoint URL but the
/// deployment never configured `foundation.function_endpoint_url`. Surfaced
/// as a 500 so a misconfigured deploy fails loudly instead of attaching the
/// function with a missing/placeholder endpoint.
#[derive(Debug, thiserror::Error)]
#[error("foundation.function_endpoint_url is not configured")]
struct MissingFunctionEndpointUrl;

impl ChromaError for MissingFunctionEndpointUrl {
    fn code(&self) -> ErrorCodes {
        ErrorCodes::Internal
    }
}

/// Idempotently create the shared foundation function on the base source
/// collection. Additional source collections are attached later via
/// `add_input()`. Params carry the modal endpoint and the base source kind,
/// matching the existing Foundation function contract.
///
/// `/init` is safe to call repeatedly — the shared helper treats an
/// already-existing function as a no-op (`created = false`).
async fn ensure_attached_function(
    sysdb: &mut SysDb,
    tenant: String,
    input_collection_id: CollectionUuid,
    base_source_name: &str,
    cfg: &FoundationConfig,
) -> Result<(), ServerError> {
    let endpoint_url = cfg
        .function_endpoint_url
        .as_ref()
        .ok_or(MissingFunctionEndpointUrl)?;
    let source_kind = source_kind_for_collection_name(base_source_name)?;
    let params = serde_json::json!({
        "endpoint_url": endpoint_url,
        "source_collection": base_source_name,
        "source_kind": source_kind,
    });
    let output_schema = Schema::new_record_only();
    attached_function_ops::create_attached_function(
        sysdb,
        foundation_attached_function_name(),
        cfg.function_name.clone(),
        input_collection_id,
        cfg.wiki_collection.clone(),
        params,
        tenant,
        cfg.database_name.clone(),
        cfg.min_records_for_invocation,
        output_schema,
    )
    .await?;
    Ok(())
}

fn foundation_attached_function_name() -> String {
    "foundation_sources_to_wiki".to_string()
}

/// Attach the built-in `revision_history` function to the wiki
/// collection so every upsert/delete is archived into the wiki_revisions
/// collection on compaction.
async fn ensure_revision_history_function(
    sysdb: &mut SysDb,
    tenant: String,
    wiki_collection_id: CollectionUuid,
    cfg: &FoundationConfig,
) -> Result<(), ServerError> {
    let params = serde_json::json!({
        "version_key": "version",
    });
    let output_schema = Schema::new_record_only();
    attached_function_ops::create_attached_function(
        sysdb,
        "wiki_revision_history".to_string(),
        "revision_history".to_string(),
        wiki_collection_id,
        cfg.wiki_revisions_collection.clone(),
        params,
        tenant,
        cfg.database_name.clone(),
        cfg.min_records_for_invocation,
        output_schema,
    )
    .await?;
    Ok(())
}

/// Attach the configured wiki->currents function to the wiki collection so
/// currents are refreshed whenever the wiki advances.
async fn ensure_currents_function(
    sysdb: &mut SysDb,
    tenant: String,
    wiki_collection_id: CollectionUuid,
    cfg: &FoundationConfig,
) -> Result<(), ServerError> {
    let endpoint_url = cfg
        .function_endpoint_url
        .as_ref()
        .ok_or(MissingFunctionEndpointUrl)?;
    let params = serde_json::json!({
        "endpoint_url": endpoint_url,
        "database_name": cfg.database_name,
    });
    let output_schema = Schema::new_record_only();
    attached_function_ops::create_attached_function(
        sysdb,
        foundation_currents_attached_function_name(),
        cfg.currents_function_name.clone(),
        wiki_collection_id,
        cfg.currents_collection.clone(),
        params,
        tenant,
        cfg.database_name.clone(),
        cfg.min_records_for_invocation,
        output_schema,
    )
    .await?;
    Ok(())
}

fn foundation_currents_attached_function_name() -> String {
    "wiki_currents".to_string()
}

#[derive(Debug, thiserror::Error)]
enum FoundationInitError {
    #[error("Configured foundation database name is shorter than the 3-character minimum")]
    DatabaseNameTooShort,
}

impl ChromaError for FoundationInitError {
    fn code(&self) -> ErrorCodes {
        match self {
            FoundationInitError::DatabaseNameTooShort => ErrorCodes::InvalidArgument,
        }
    }
}

async fn ensure_currents_collection(
    sysdb: &mut SysDb,
    tenant: String,
    db_name: DatabaseName,
    cfg: &FoundationConfig,
) -> Result<Collection, ServerError> {
    ensure_collection(
        sysdb,
        tenant,
        db_name,
        &cfg.currents_collection,
        None,
        Some(1),
        CollectionEmbeddingFunctions::default(),
    )
    .await
}

#[tracing::instrument(
    name = "ensure_database",
    skip_all,
    fields(database = %database_name.as_ref(), tenant = %tenant),
    err(Display)
)]
async fn ensure_database(
    sysdb: &mut SysDb,
    database_name: DatabaseName,
    tenant: String,
) -> Result<Uuid, ServerError> {
    // Generate the id once so retries don't churn through fresh UUIDs; the
    // create is keyed on the (tenant, name) pair and is idempotent — an
    // `AlreadyExists` from a previous/concurrent init is the success path.
    let database_id = Uuid::new_v4();
    let created = match retry_transient(|| {
        let mut sysdb = sysdb.clone();
        let database_name = database_name.clone();
        let tenant = tenant.clone();
        async move {
            sysdb
                .create_database(database_id, database_name, tenant)
                .await
        }
    })
    .await
    {
        Ok(_) => true,
        Err(CreateDatabaseError::AlreadyExists(_)) => false,
        Err(e) => return Err(e.into()),
    };

    let db = retry_transient(|| {
        let mut sysdb = sysdb.clone();
        let database_name = database_name.clone();
        let tenant = tenant.clone();
        async move { sysdb.get_database(database_name, tenant).await }
    })
    .await?;

    tracing::info!(database = %database_name.as_ref(), database_id = %db.id, created, "ensured foundation database");
    Ok(db.id)
}

/// SysDb's `create_collection` takes a `get_or_create: bool`. When true, an
/// existing collection with the same (tenant, database, name) is returned
/// instead of failing with `AlreadyExists` — atomic idempotency in one round
/// trip, so we don't need the try-then-fallback dance we use for databases.
const GET_OR_CREATE: bool = true;

/// Plan a fresh distributed-mode collection with the shared
/// `frontend_core::collection_ops` planner and hand it to sysdb. The
/// planner reconciles the Foundation schema (sparse vector index for
/// SPLADE) with the default config, picks the right segment types for
/// distributed mode, and emits everything sysdb needs.
#[tracing::instrument(
    name = "ensure_collection",
    skip_all,
    fields(collection = %collection_name, database = %database_name.as_ref()),
    err(Display)
)]
async fn ensure_collection(
    sysdb: &mut SysDb,
    tenant: String,
    database_name: DatabaseName,
    collection_name: &str,
    metadata: Option<Metadata>,
    dimension: Option<i32>,
    embedding_functions: CollectionEmbeddingFunctions,
) -> Result<Collection, ServerError> {
    let schema = foundation_collection_schema(embedding_functions);
    create_planned_collection(
        sysdb,
        tenant,
        database_name,
        collection_name,
        schema,
        metadata,
        dimension,
    )
    .await
}

/// Ensure an UNINDEXED, record-only collection (e.g. `slack_raw`). Mirrors
/// [`ensure_collection`] but uses the shared record-only schema, no embedding
/// function, and no pinned dimension — records are stored verbatim and never
/// indexed at ingest. The schema spec is shared with hosted-chroma's sync
/// service via [`chroma_types::slack_raw_schema`].
#[tracing::instrument(
    name = "ensure_record_only_collection",
    skip_all,
    fields(collection = %collection_name, database = %database_name.as_ref()),
    err(Display)
)]
async fn ensure_record_only_collection(
    sysdb: &mut SysDb,
    tenant: String,
    database_name: DatabaseName,
    collection_name: &str,
    metadata: Option<Metadata>,
) -> Result<Collection, ServerError> {
    create_planned_collection(
        sysdb,
        tenant,
        database_name,
        collection_name,
        slack_raw_schema(),
        metadata,
        // Record-only: no dense vectors are written, so there is no dimension
        // to pin (mirrors currents / wiki_revisions, but with all indexing off).
        None,
    )
    .await
}

/// Shared core for the `ensure_*_collection` helpers: plan a fresh
/// distributed-mode collection from the given `schema` and hand it to sysdb.
/// `GET_OR_CREATE` keeps it idempotent in a single round trip, so a transient
/// sysdb failure is safe to retry; the plan (collection id + segments +
/// config) is fixed up front and reused across attempts.
async fn create_planned_collection(
    sysdb: &mut SysDb,
    tenant: String,
    database_name: DatabaseName,
    collection_name: &str,
    schema: Schema,
    metadata: Option<Metadata>,
    dimension: Option<i32>,
) -> Result<Collection, ServerError> {
    let plan = plan_create_collection(
        None,
        Some(schema),
        ExecutorKind::Distributed,
        &supported_segment_types(ExecutorKind::Distributed),
        true,
        KnnIndex::Spann,
        TenantFeatureFlags::default(),
    )?;
    let collection_id = plan.collection_id;
    let collection = retry_transient(|| {
        let mut sysdb = sysdb.clone();
        let tenant = tenant.clone();
        let database_name = database_name.clone();
        let collection_name = collection_name.to_string();
        let segments = plan.segments.clone();
        let configuration = plan.configuration.clone();
        let schema = plan.schema.clone();
        let metadata = metadata.clone();
        async move {
            sysdb
                .create_collection(
                    tenant,
                    database_name,
                    collection_id,
                    collection_name,
                    segments,
                    configuration,
                    schema,
                    metadata,
                    dimension,
                    GET_OR_CREATE,
                )
                .await
        }
    })
    .await?;

    tracing::info!(
        collection = %collection_name,
        collection_id = %collection.collection_id,
        "ensured foundation collection"
    );
    Ok(collection)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gdrive_source_is_single_dimension_others_are_1024() {
        assert_eq!(source_dimension("gdrive"), Some(1));
        assert_eq!(source_dimension("gdrive_master"), Some(1));
        assert_eq!(source_dimension("notion"), Some(1024));
        // Unknown sources fall back to the default 1024 dims.
        assert_eq!(source_dimension("unknown_source"), Some(1024));
    }

    /// `slack_raw` is the attached function's base input, so its source_kind
    /// must resolve to `slack` — that keeps the generation contract identical
    /// to the old `slack` base and guarantees `ensure_attached_function` won't
    /// error on an unknown source kind.
    #[test]
    fn slack_raw_maps_to_slack_source_kind() {
        assert_eq!(
            source_kind_for_collection_name(SLACK_RAW_COLLECTION_NAME).unwrap(),
            "slack"
        );
    }

    /// `slack_raw` is created through the same planner path `/init` uses for
    /// the record-only helper. Assert the reconciled plan keeps every index
    /// disabled — i.e. the collection really is record-only after planning,
    /// not just before `reconcile_schema_and_config` runs.
    #[test]
    fn slack_raw_plan_is_record_only() {
        let plan = plan_create_collection(
            None,
            Some(slack_raw_schema()),
            ExecutorKind::Distributed,
            &supported_segment_types(ExecutorKind::Distributed),
            true,
            KnnIndex::Spann,
            TenantFeatureFlags::default(),
        )
        .expect("planning the record-only slack_raw schema must succeed");

        let reconciled = plan
            .schema
            .as_ref()
            .expect("plan must carry a reconciled schema when enable_schema=true");

        assert!(
            !reconciled.is_sparse_index_enabled(),
            "slack_raw must have no sparse vector index after planning"
        );
        assert!(
            !reconciled.is_fts_enabled(),
            "slack_raw must have no FTS index after planning"
        );

        let float_list = reconciled
            .defaults
            .float_list
            .as_ref()
            .expect("schema defaults must carry a dense vector index entry");
        assert!(
            !float_list.vector_index.as_ref().unwrap().enabled,
            "slack_raw must have no dense vector index after planning"
        );

        let string = reconciled
            .defaults
            .string
            .as_ref()
            .expect("schema defaults must carry a string value type");
        assert!(
            !string.string_inverted_index.as_ref().unwrap().enabled,
            "slack_raw must have no string inverted index after planning"
        );
    }
}
