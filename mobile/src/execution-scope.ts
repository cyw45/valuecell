export const ALL_EXECUTION_BATCHES = "__all__";

export type ExecutionQueryScope = {
  ready: boolean;
  batchId: string | null;
  allHistory: boolean;
  unavailableReason: "batch_pending" | "no_current_batch" | null;
};

type ExecutionScopeInput = {
  environment?: string | null;
  status?: string | null;
  /** undefined means the batch list has not loaded; null means it loaded with no current batch. */
  currentBatchId: string | null | undefined;
  selectedBatchId?: string | null;
};

type ReasonOrder = {
  id?: string | null;
  decision_reason?: string | null;
  decision_reason_code?: string | null;
};

type ReasonFact = {
  order_id: string | null;
  explanation?: {
    decision?: string | null;
    decision_reason?: string | null;
  } | null;
};

/** Match the web trades page: a stopped Demo strategy hides every order when batch_id is omitted. */
export function executionQueryScope(input: ExecutionScopeInput): ExecutionQueryScope {
  if (input.selectedBatchId === ALL_EXECUTION_BATCHES) {
    return { ready: true, batchId: null, allHistory: true, unavailableReason: null };
  }
  if (input.selectedBatchId) {
    return { ready: true, batchId: input.selectedBatchId, allHistory: false, unavailableReason: null };
  }
  const stoppedDemo = input.environment === "okx_demo" && input.status !== "running";
  if (stoppedDemo && input.currentBatchId === undefined) {
    return { ready: false, batchId: null, allHistory: false, unavailableReason: "batch_pending" };
  }
  if (stoppedDemo && input.currentBatchId === null) {
    return { ready: true, batchId: null, allHistory: false, unavailableReason: "no_current_batch" };
  }
  return {
    ready: true,
    batchId: input.currentBatchId ?? null,
    allHistory: false,
    unavailableReason: null,
  };
}

export function attributedDecisionReason(
  order: ReasonOrder,
  facts: readonly ReasonFact[],
): string {
  const direct = order.decision_reason?.trim();
  if (direct) return direct;
  const fact = facts.find((item) => item.order_id != null && item.order_id === order.id);
  const fromFact = fact?.explanation?.decision_reason?.trim();
  if (fromFact) return fromFact;
  const decision = fact?.explanation?.decision?.trim();
  if (decision) return decision;
  const code = order.decision_reason_code?.trim();
  if (code) return code;
  return "服务端未提供成交原因。";
}
