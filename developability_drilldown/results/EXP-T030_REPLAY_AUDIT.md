# EXP-T030 replay audit

- git: `4a9dccb27fd60fabe36b828d6f7010feb93f75c2`
- run_id: `EXP-T030-REPLAY-001`
- HISTORICAL_REPRODUCTION: **FAIL**
- CONTEMPORARY_MATCHED_CONTROL: **ACCEPTED**

## Historical canonical EXP-T030 (unchanged)
- Primary: 3.317769289998
- Shadow: 3.214610237153
- reproducibility_status: RESULT_VERIFIED (not promoted to REPRODUCED)

## Contemporary matched control (EXP-T030-REPLAY-001)
- Primary: 3.232677420471
- Shadow: 3.201957977358
- Public: 3.595528851852
- Private: 3.255966469136
- Overall: 3.425747660494
- primary max |Δ| vs historical OOF: 2.741976e+00
- shadow max |Δ| vs historical OOF: 3.205467e+00

## Conclusion

Historical T030 is NOT reproduced under the current environment.
The contemporary run is accepted as the matched scientific control for T068/T069.
Historical predictions/scores/config remain untouched.

