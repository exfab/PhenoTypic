# Round 3 — concurrency specialist (in progress)

Scope: snapshots/round-2-spec.diff (1367 lines) + round-2-plan.diff (103).
Job: verify the concurrency block landed as decided; check propagation.

## Working notes

## Verification sweep (diff read end to end)

Fidelity of the concurrency block vs the round-2 decision list: HIGH.
Verified present and correct:
- asyncio.Semaphore(1) owned by the loop; call_soon_threadsafe for thread/process side (01:354-397)
- release in a finally at the innermost acquiring layer; exit observer release-first/record-second with the record step in its own try/except
- wall-clock lease, unconditional, with the "regardless of how cancellation tests out" framing intact
- (pid, create_time) identity, both in 1.5 and 2.4; refuse-not-watch with local_slot_orphaned
- start_new_session=True detach (USER-23)
- CAS on (status, artifact_digest); digest captured when the prompt is BUILT; campaign_changed_during_approval
- launching->running never over a terminal status
- file lock before threading.Lock, with the runs.py:317-337 citation
- _REGISTRY published after discover() under a module lock
- staging temp -> os.replace -> .complete marker, readers require marker
- probe output keyed (pipeline digest, image-set digest, uuid4)
- cursor absent state + study.db-wal
- server-wide max_inflight_arms=8 checked by the launcher; "sbatch accepted it proves nothing"
- 7 new error codes incl. artifact_lock_timeout + output_generation_active
- bounds table in 1.6.1
- test list covers release-on-4-paths, lease expiry, executor isolation, pid-reuse, staging marker, fan-out idempotency, cursor states

## Defects found (detail in the sent report)
- CONC-29 Critical: deploy_start has no human_response param; 5.4 still says ack is recorded ON THE TOKEN (USER-18 deleted that); plan D6 still "required-unless-elicited"
- CONC-30 Major: local_slot_capacity configurable vs compute=1 fixed vs "one W1 in flight" invariant; contention not disclosed in probe timings
- CONC-31 Major: 1.5 compute pool restates in-process run_in_executor W1 that 3.2 refutes (probe is a subprocess); which pool carries the probe wait is undefined
- CONC-32 Major: 8.7 two-part row requires mutating an append-only journal; no step_id; example row still shows persisted decision:"keep"; in_flight has no expiry
- CONC-33 Major: N per-campaign launchers contend for one server-wide ceiling with no arbitration (USER-24 fan-out shape)
