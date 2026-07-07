# ATC HMI Visual Demo and LLM Wrapper Design

## Goal

This demo shows a human-machine ATC decision loop in BlueSky:

1. A dynamic sector spawns up to 14 aircraft on crossing routes.
2. The monitor checks every aircraft pair for predicted separation risk.
3. A safety-constrained solver searches altitude and speed actions.
4. An LLM-style wrapper turns the verified action plan into structured fields, standard controller instructions, and a short rationale.
5. The GUI writes the conflict, solver decision, BlueSky commands, and explanation into a JSONL log.

The LLM layer is deliberately downstream of the safety verifier. It can select or express preferences, but it does not bypass the verified action constraints.

## Visual Interface

The BlueSky bottom `AI` tab contains:

- `Reset sector`: clears traffic and loads the sector viewport.
- `Start auto traffic`: spawns the initial traffic wave, starts random boundary spawning, starts conflict monitoring, and switches BlueSky to operate mode.
- `Stop`: pauses the demo.
- `Spawn one`: injects one random aircraft from a route boundary.
- `Detect now`: runs conflict detection and resolution immediately.
- `Fast 2 min`: advances the simulation by two minutes.
- `Preference`: switches the solver action ordering between `speed_first` and `altitude_first`.
- `LLM wrapper`: supports `template_explainer`, `openai_compatible_api`, and `off`. `template_explainer` is a deterministic local wrapper that mirrors the expected LLM output contract without requiring network access. `openai_compatible_api` calls an optional chat-completions-compatible endpoint.

The decision table records:

- time,
- event type,
- involved aircraft,
- CPA estimate,
- decision or intent preference,
- issued BlueSky command,
- verification status.

The status row shows:

- active aircraft count,
- conflict-detection cycle count,
- cumulative conflict events,
- issued command count,
- current LLM wrapper status,
- current JSONL log filename.

The text pane shows the standard instruction phrase and concise safety rationale.

## Solver Input

Each detection cycle reads live BlueSky `ACDATA`:

- callsign,
- latitude and longitude,
- altitude,
- track,
- ground speed.

For every aircraft pair, the GUI computes closest point of approach over a 12-minute lookahead. A pair enters the conflict graph when predicted horizontal proximity is inside the prediction gate and the current target plan is not safe under forward verification.

In the GUI, the preferred state source is live BlueSky `ACDATA`. For short demonstrations, QtGL stream delivery can be sparse while commands are still being processed. To keep the human-machine workflow visible, the panel includes a demonstration fallback that reconstructs aircraft states from the route, speed, heading, flight level, and spawn time recorded by the AI panel. This fallback is explicitly for interface demonstration; the headless validation script remains the source of quantitative safety evidence.

## Safety-Constrained Action Search

The GUI now uses a discrete constraint-search solver aligned with the headless validation logic.

Candidate actions per aircraft:

- hold current target,
- altitude change by 4000 ft within FL270-FL430,
- speed change by -40, -20, +20, or +40 kt within 380-500 kt.

Forward verification:

- lookahead horizon: 12 minutes,
- prediction step: 2 seconds,
- horizontal verification threshold: 7 NM,
- vertical verification threshold: 2000 ft,
- climb/descent rate: 2000 ft/min,
- speed acceleration: 1 kt/s.

The solver searches the conflict graph and only issues actions that pass pairwise forward verification. If no verified plan is found, the GUI logs the conflict and does not issue an unverified fallback command.

## LLM Wrapper Contract

The current GUI implements a local deterministic wrapper with the same output shape expected from a future LLM call:

```json
{
  "provider": "template_explainer",
  "prompt_contract": "conflict_state + controller_preference -> structured_actions + standard_phrase + rationale",
  "preference": "speed_first",
  "conflicts": [],
  "structured_actions": [],
  "standard_instructions": [],
  "explanation": ""
}
```

This is suitable for demo and testing because it avoids network/API dependency while preserving the interface boundary. A real LLM can be enabled without changing conflict detection and safety verification.

Optional real-model environment variables:

```text
ATC_LLM_API_URL=http://127.0.0.1:8000/v1/chat/completions
ATC_LLM_MODEL=qwen3-4b
ATC_LLM_API_KEY=optional_key
```

When `openai_compatible_api` is selected, the GUI sends the verified decision payload to the configured endpoint and uses the returned text as the explanation. If the API is unavailable, the GUI records the API error and still keeps the verified local decision output.

For offline demonstration, run:

```powershell
.\RUN_MOCK_LLM.ps1
```

This starts a local deterministic `/v1/chat/completions` endpoint. It is not a real model, but it verifies the same request/response boundary used by an OpenAI-compatible LLM service.

## Runtime Logs

GUI logs are written under:

```text
bluesky_project/output/hmi_dynamic_logs/
```

Each conflict event records:

- detected conflict pairs,
- CPA metrics,
- solver method,
- selected preference,
- selected actions,
- issued BlueSky commands,
- LLM-wrapper structured output and explanation.

## Demo Script

1. Start BlueSky with `python BlueSky.py`.
2. Open the bottom `AI` tab.
3. Click `Reset sector`.
4. Select `speed_first` or `altitude_first`.
5. Keep `LLM wrapper` as `template_explainer`, or switch to `openai_compatible_api` after setting the environment variables.
6. Click `Start auto traffic`.
7. Watch aircraft spawn on crossing routes.
8. When a conflict is detected, show:
   - the involved aircraft,
   - the verified speed/altitude command,
   - the standard instruction phrase,
   - the safety rationale,
   - the JSONL log entry.
