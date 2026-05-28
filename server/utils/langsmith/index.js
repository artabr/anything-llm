const { Client } = require("langsmith");
const { AsyncLocalStorage } = require("async_hooks");
const { randomUUID } = require("crypto");

let _client = null;
const _traceStorage = new AsyncLocalStorage();

function getClient() {
  if (!process.env.LANGSMITH_API_KEY) return null;
  if (!_client) _client = new Client({ apiKey: process.env.LANGSMITH_API_KEY });
  return _client;
}

function isEnabled() {
  return !!process.env.LANGSMITH_API_KEY;
}

function getProjectName() {
  return process.env.LANGSMITH_PROJECT || "AnythingLLM";
}

/**
 * Returns the current trace context from AsyncLocalStorage, or null.
 * @returns {{ id: string, projectName: string }|null}
 */
function getTraceContext() {
  return _traceStorage.getStore() || null;
}

/**
 * Runs fn inside an AsyncLocalStorage context so all nested async calls
 * can access traceContext via getTraceContext(). No-op if traceContext is null.
 * @param {{ id: string, projectName: string }|null} traceContext
 * @param {Function} fn
 */
function runWithTraceContext(traceContext, fn) {
  if (!traceContext) return fn();
  return _traceStorage.run(traceContext, fn);
}

/**
 * Creates a parent "chain" run for the entire chat request. Fire-and-forget.
 * Returns a traceContext synchronously so it can be used immediately.
 * Pass externalParentRunId (from x-langsmith-trace-id header) to nest under a caller's run.
 * @param {{ message: string, workspaceSlug: string, externalParentRunId?: string|null }} opts
 * @returns {{ id: string, projectName: string }|null}
 */
function startChatTrace({ message, workspaceSlug, externalParentRunId = null }) {
  const client = getClient();
  if (!client) return null;

  const id = randomUUID();
  const projectName = getProjectName();
  const startTime = Date.now();

  (async () => {
    try {
      await client.createRun({
        id,
        name: `chat: ${workspaceSlug}`,
        run_type: "chain",
        project_name: projectName,
        inputs: { message },
        start_time: startTime,
        ...(externalParentRunId ? { parent_run_id: externalParentRunId } : {}),
      });
    } catch (e) {
      console.debug(`[LangSmith] Failed to create chat trace: ${e.message}`);
    }
  })();

  return { id, projectName };
}

/**
 * Finalizes a parent chat run with outputs. Fire-and-forget.
 * @param {{ id: string, projectName: string }|null} traceContext
 * @param {{ textResponse?: string, sources?: Array, metrics?: Object }} opts
 */
function endChatTrace(
  traceContext,
  { textResponse = "", sources = [], metrics = {} } = {}
) {
  if (!traceContext) return;
  const client = getClient();
  if (!client) return;

  (async () => {
    try {
      await client.updateRun(traceContext.id, {
        outputs: { text: textResponse, sources_count: sources.length },
        end_time: Date.now(),
        extra: { metrics },
      });
    } catch (e) {
      console.debug(`[LangSmith] Failed to end chat trace: ${e.message}`);
    }
  })();
}

/**
 * Traces a completed LLM call. Fire-and-forget.
 * Automatically links to the current trace context as a child run when available.
 * @param {Object} opts
 * @param {string} opts.runName
 * @param {string} opts.provider
 * @param {string} opts.model
 * @param {Array} opts.messages
 * @param {string} opts.textResponse
 * @param {Object} opts.metrics
 */
function traceLLMCall({
  runName,
  provider,
  model,
  messages,
  functions,
  textResponse,
  metrics,
}) {
  const client = getClient();
  if (!client) return;

  const traceContext = getTraceContext();
  const projectName = traceContext?.projectName || getProjectName();
  const endTime = metrics?.timestamp || new Date();
  const startTime = metrics?.duration
    ? new Date(endTime.getTime() - Math.round(metrics.duration * 1000))
    : endTime;

  (async () => {
    try {
      await client.createRun({
        id: randomUUID(),
        name: runName || `${provider} / ${model}`,
        run_type: "llm",
        project_name: projectName,
        inputs: {
          messages,
          ...(Array.isArray(functions) && functions.length > 0
            ? { tools: functions }
            : {}),
        },
        outputs: { text: textResponse },
        start_time: startTime.getTime(),
        end_time: endTime.getTime(),
        ...(traceContext?.id ? { parent_run_id: traceContext.id } : {}),
        extra: {
          model,
          provider,
          prompt_tokens: metrics?.prompt_tokens ?? 0,
          completion_tokens: metrics?.completion_tokens ?? 0,
          total_tokens: metrics?.total_tokens ?? 0,
          output_tps: metrics?.outputTps ?? 0,
          duration_seconds: metrics?.duration ?? 0,
        },
      });
    } catch (e) {
      console.debug(`[LangSmith] Failed to trace run: ${e.message}`);
    }
  })();
}

/**
 * Traces a vector DB similarity search as a retriever child run.
 * No-op if there is no active trace context.
 * @param {{ query: string, namespace: string, results?: Array, duration?: number }} opts
 */
function traceRetrieval({ query, namespace, results = [], duration = 0 }) {
  const client = getClient();
  if (!client) return;
  const traceContext = getTraceContext();
  if (!traceContext?.id) return;

  const projectName = traceContext.projectName || getProjectName();
  const endTime = Date.now();
  const startTime = endTime - Math.round(duration * 1000);

  (async () => {
    try {
      await client.createRun({
        id: randomUUID(),
        name: `retrieval: ${namespace}`,
        run_type: "retriever",
        project_name: projectName,
        parent_run_id: traceContext.id,
        inputs: { query },
        outputs: {
          documents: results.map((r) => ({
            pageContent: r.text || r.pageContent || "",
            metadata: { title: r.title, score: r.score },
          })),
        },
        start_time: startTime,
        end_time: endTime,
      });
    } catch (e) {
      console.debug(`[LangSmith] Failed to trace retrieval: ${e.message}`);
    }
  })();
}

/**
 * Traces an agent tool invocation as a tool child run.
 * No-op if there is no active trace context.
 * @param {{ toolName: string, args?: Object, result?: any, duration?: number, error?: string|null }} opts
 */
function traceToolCall({ toolName, args = {}, result = null, duration = 0, error = null }) {
  const client = getClient();
  if (!client) return;
  const traceContext = getTraceContext();
  if (!traceContext?.id) return;

  const projectName = traceContext.projectName || getProjectName();
  const endTime = Date.now();
  const startTime = endTime - Math.round(duration * 1000);

  (async () => {
    try {
      await client.createRun({
        id: randomUUID(),
        name: `tool: ${toolName}`,
        run_type: "tool",
        project_name: projectName,
        parent_run_id: traceContext.id,
        inputs: { args },
        outputs: error ? { error } : { result },
        start_time: startTime,
        end_time: endTime,
        error: error || undefined,
      });
    } catch (e) {
      console.debug(`[LangSmith] Failed to trace tool call: ${e.message}`);
    }
  })();
}

module.exports = {
  traceLLMCall,
  isEnabled,
  startChatTrace,
  endChatTrace,
  runWithTraceContext,
  getTraceContext,
  traceRetrieval,
  traceToolCall,
};
