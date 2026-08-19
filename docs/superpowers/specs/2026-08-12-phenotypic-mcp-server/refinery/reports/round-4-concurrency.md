# Round 4 — concurrency specialist (in progress)
Scope: snapshots/round-3-spec.diff (1459) + round-3-plan.diff.
Job: verify CONC-29..33 closed in the DEFINING sections, not prose.

## CONC-29..33 verification: 4.5 of 5 closed in DEFINING sections

- CONC-29 CLOSED (over-closed, in a good way): 05:§5.4 arg table has human_response (Required) + note?;
  response carries ack_source; ordering stated "elicit -> allocate -> acquire"; token persists
  ack_prompt + decision_content so the server renders the prompt; binding table with a Scope column;
  new "two producers" table (kind: plan | campaign_arm). Plan D6 updated.
- CONC-30 PARTIAL: 06:155,157 fixed (compute = local_slot_capacity); 02:330 fixed;
  01:237 fixed; 01:447 Semaphore(local_slot_capacity) fixed.
  *** 01-architecture.md:604 STILL SAYS "1" with the retired "second expression" rationale ***
  -> that is the 1.6.1 stated-bounds table, a defining section by their own map.
- CONC-31 CLOSED: 01:413-430 rewritten; compute = probe-dispatch slot, not a compute pool;
  residency claim corrected to the subprocess.
- CONC-32 CLOSED: two appends correlated by step_id; in_flight lease; example row fixed;
  per-kind match key + derivation table (6 kinds, remove_op inverted, undetermined for move_op
  and ambiguous set_params); attribution limit stated.
- CONC-33 CLOSED: asyncio.Semaphore(max_inflight_arms), arrival order, wake condition = the
  semaphore itself, cancel/shutdown behaviour, queued_reason x3 incl. local_slot, launcher lease
  {pid, create_time, expires}, two-armed transition with launch_state as discriminator,
  write_generation demoted to a read hint. Fields declared in 8.2 schema.

## New this round
- CONC-34 Critical: deploy_start requires human_response unconditionally, but 10.4 carries an
  UNATTENDED deploy arm whose token is kind=campaign_arm. No human at 3am -> launcher must
  fabricate the field, and ack_source would read agent_asserted although a human DID approve
  at campaign_approve. Fix: require human_response when token kind == "plan"; campaign_arm
  carries consent forward with ack_source: "campaign_approved". Keyed on the token the caller
  passes, NOT on host capability, so USER-22 is not reopened.
- CONC-35 Major: 01:604 bounds row.
- CONC-36 Major: USER-26 manifest absent from 5.4 entirely - no binding row, no record field,
  no storage path, no collection rule. argv_digest binds the argv NAMING the manifest, not its
  contents. 10.5 claims "the manifest is what the token's digest binds"; nothing implements that.
