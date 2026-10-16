# Replay fixture provenance (OEV-001)

`sample02_run34236809938_f0000-1299.jsonl.gz`

- Source: `events.jsonl` of OEV GPU run 34236809938 (sample_02, 180 s,
  YOLO26m, stride 1), produced by video-stitcher `fa3c79e2` (ROI +0.40
  experiment). Full cache: Drive file `1qAMhkVjN3NhgDpysIKEb6IgQ07jkmxpe`,
  71,395,983 bytes, sha256
  `963ede96c18bf9dec50ec6788da38be38381bfade5e86b2ea0ad775f2f3ef204`.
  Large caches stay outside Git; manifest in ffa-automations
  `oev/cpu_replay/provenance.json`.
- Contents: frames 0-1299, in recorded order, per frame
  `frame_start` (index only), `detections_raw` filtered to ball class 32,
  and the unmodified `world_state` (timestamp, players, ball). Other
  detection classes are dropped because `BallTracker` ignores them on every
  path (update, frame context, reactivation all filter by `class_id`).
- `detections_raw` is the post-wrapper mapped tracker input (ROI-filtered,
  projected). It is not a pre-ROI detector cache.
- `world_state.timestamp_ms` is the exact tracker input time. At stride 1
  it is wall-clock processing time (~220 ms/frame here), not video time.
- `world_state.ball` is post backward-bridge.
- Covers: a dormant transition (frame 665), an actual release (742) and a
  second dormancy at the decoy-adjacent site (1202).
- sha256 (gz): `d7c7c23da805bcc560bf57c01886b27f9f6b2940ef8ddc0d285639bef18f37f5`
- sha256 (uncompressed): `9dba34b9f714aaba7af99ad318e52b4ec9c7fae87dc5e350ae33bcbbc9297ef8`
