# GigaChat Integration Notes (Thread Summary)

## Goal

Add first-class GigaChat provider support to Langfuse (not only OpenAI-compatible custom base URL usage), and document pricing assumptions for now.

## Integration Approach Chosen

- Implemented as dedicated adapter: `LLMAdapter.GigaChat` (`"gigachat"`).
- Added provider-specific config and defaults:
  - `GIGACHAT_DEFAULT_SCOPE = "GIGACHAT_API_PERS"`
  - `GIGACHAT_DEFAULT_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1/"`
- For generic completions/tool-calls, still uses OpenAI-compatible model invocation path after OAuth token exchange.
- For evaluator structured output (LLMAJ), now uses a dedicated native GigaChat function-calling request path (`functions` + forced `function_call`).

## Inputs Provided During Thread

- LLM connection secret semantics:
  - `GIGACHAT_CREDENTIALS`
  - `GIGACHAT_SCOPE` (default `GIGACHAT_API_PERS`)
  - `GIGACHAT_BASE_URL` (default `https://gigachat.devices.sberbank.ru/api/v1/`)
- Structured output compatibility:
  - GigaChat structured output is compatible.
  - Default method is function calling; json mode exists as well.
- Source links:
  - Models endpoint: `https://gigachat.devices.sberbank.ru/api/v1/models`
  - Pricing page: `https://developers.sber.ru/docs/ru/gigachat/tariffs/individual-tariffs#platnye-pakety`

## Files Changed

### Shared (`packages/shared`)

- `packages/shared/src/interfaces/customLLMProviderConfigSchemas.ts`
  - Added `GigaChatConfigSchema`.
  - Added `GIGACHAT_DEFAULT_SCOPE`.
  - Added `GIGACHAT_DEFAULT_BASE_URL`.

- `packages/shared/src/server/llm/types.ts`
  - Added `LLMAdapter.GigaChat`.
  - Added `gigaChatModels`:
    - `GigaChat-2`
    - `GigaChat-2-Max`
    - `GigaChat-2-Pro`
    - `GigaChat-2-Lite`
  - Added GigaChat to `supportedModels`.
  - Extended `LLMApiKeySchema.config` union with `GigaChatConfigSchema`.

- `packages/shared/src/server/llm/fetchLLMCompletion.ts`
  - Added GigaChat execution branch.
  - Added OAuth token exchange helper:
    - POST to `/api/v2/oauth`
    - `Authorization: Basic <credentials>`
    - body includes `scope`
    - includes `RqUID` header
  - Uses returned access token for chat completion via OpenAI-compatible `ChatOpenAI`.
  - Added GigaChat-native structured-output path for LLMAJ:
    - Sends `POST /chat/completions` with `functions` and forced `function_call`.
    - Converts eval Zod schema to JSON schema for function `parameters`.
    - Parses response from `choices[0].message.function_call.arguments` (with content fallback).
  - Added GigaChat OAuth token cache + in-flight dedupe:
    - Reuses token until near expiry.
    - Prevents many parallel eval jobs from requesting a new token simultaneously.
    - Handles `expires_at` / `expires_in` when present; otherwise uses safe fallback TTL.

### Web (`web`)

- `web/src/features/llm-api-key/types.ts`
  - Added GigaChat config schema to connection payload union.

- `web/src/features/llm-api-key/server/router.ts`
  - Extended `testLLMConnection` config parsing for GigaChat (`scope`).

- `web/src/features/public-api/types/llm-connections.ts`
  - Added GigaChat config validation in `PutLlmConnectionV1Body`.
  - Allows optional GigaChat config (`{ scope: string }`).

- `web/src/features/public-api/components/CreateLLMApiKeyForm.tsx`
  - Added GigaChat-aware form behavior:
    - Base URL default and required validation.
    - Scope field in advanced settings.
    - Secret label changed to `GigaChat Credentials` for this adapter.
    - Help text: credentials can be provided with or without `Basic ` prefix.
  - Included adapter in advanced settings handling.

- `web/src/features/playground/page/hooks/useModelParams.ts`
  - Added default parameter branch for `LLMAdapter.GigaChat`.

- `web/src/__tests__/server/llm-connections-api.servertest.ts`
  - Added GigaChat in "all adapters" coverage.
  - Added config validation case for GigaChat.

### Worker (`worker`)

- `worker/src/__tests__/llmConnections.test.ts`
  - Added GigaChat integration test scaffold and env var docs.
  - Added GigaChat evaluator structured-output coverage via shared eval test helper.

- `worker/src/constants/default-model-prices.json`
  - Added approximate USD pricing entries for:
    - `gigachat-2` (also matches `gigachat`)
    - `gigachat-2-lite`
    - `gigachat-2-pro` (also matches legacy `gigachat-pro`)
    - `gigachat-2-max` (also matches legacy `gigachat-max`)

## Pricing Assumptions Used

Pricing source is package pricing in RUB, not split by input/output. To make defaults usable in Langfuse's USD/token format, approximate conversion was used for this date:

- Assumed FX: `1 USD = 90 RUB`
- Conversion from package RUB to USD/token:
  - Lite (`1300 RUB / 20,000,000`) -> `0.7222e-6`
  - Pro (`1500 RUB / 3,000,000`) -> `5.5556e-6`
  - Max (`1950 RUB / 3,000,000`) -> `7.2222e-6`
- Applied same value for `input` and `output` due to package-level pricing format.

## Validation Run Results

Passed:

- `pnpm --filter @langfuse/shared run lint`
- `pnpm --filter @langfuse/shared run typecheck`
- `pnpm --filter @langfuse/shared run build`
- `pnpm --filter web run lint`
- `pnpm --filter web run typecheck`
- `pnpm --filter worker run lint`
- `pnpm --filter worker run typecheck`
- `node .agents/skills/add-model-price/scripts/validate-pricing-file.mjs`
- Regex tests for new GigaChat price match patterns via helper script.

## Test Failure Clarification from Thread

`web` test `llm-connections-api.servertest` failing with `TypeError: fetch failed` is not due to GigaChat credentials.

Reason:

- Test helper targets `http://localhost:3000` directly.
- If web app is not reachable at port 3000, all tests in that suite fail with `fetch failed`.

How to run properly:

1. Start app stack (`pnpm run dev` or at least web + required infra).
2. Run:
   - `pnpm --filter web run test --testPathPatterns="llm-connections-api\.servertest"`

## LLMAJ Runtime Issues and Fixes (Latest)

Observed during dataset LLMAJ runs:

- Initial failure: `Model configuration not valid for evaluation. 400 status code (no body)`.
- Root cause: GigaChat structured output via generic OpenAI-style path was not reliable for evaluator schema calls.
- Fix applied: native GigaChat function-calling structured-output branch in `fetchLLMCompletion`.

Subsequent failure:

- `429 status code` with LangChain rate-limit wrapper message.
- Worker logs show concrete source:
  - `GigaChat auth failed with status 429 ...` from OAuth token endpoint (`/api/v2/oauth`), not from score schema parsing.
- Fix applied: OAuth token caching + in-flight request deduplication to reduce auth request burst load.

Current expected behavior:

- LLMAJ retries still happen via existing queue retry logic on retryable LLM errors.
- Auth 429 frequency should be significantly reduced because token is reused for ~token lifetime instead of per-call auth requests.

## Environment Variables of Interest

### For runtime GigaChat integration

- `GIGACHAT_CREDENTIALS` (used as LLM connection secret in UI/API)
- `GIGACHAT_SCOPE` (default `GIGACHAT_API_PERS`)
- `GIGACHAT_BASE_URL` (default `https://gigachat.devices.sberbank.ru/api/v1/`)

### For worker integration tests (if executing live)

- `LANGFUSE_LLM_CONNECTION_GIGACHAT_CREDENTIALS`
- `LANGFUSE_LLM_CONNECTION_GIGACHAT_BASE_URL`
- `LANGFUSE_LLM_CONNECTION_GIGACHAT_SCOPE` (optional; defaulted in test)

## Open Follow-Ups (Optional)

- Add explicit docs section in user-facing setup docs for GigaChat credentials format and defaults.
- Revisit pricing entries with a fixed FX policy or official USD-equivalent source if available.
- Decide whether to add tokenizer mapping once an appropriate tokenizer strategy is chosen for GigaChat models.

## Reference Links

- Langfuse contributing guideline:
  - https://github.com/langfuse/langfuse/blob/main/CONTRIBUTING.md
- GigaChat models:
  - https://gigachat.devices.sberbank.ru/api/v1/models
- GigaChat tariffs:
  - https://developers.sber.ru/docs/ru/gigachat/tariffs/individual-tariffs#platnye-pakety
