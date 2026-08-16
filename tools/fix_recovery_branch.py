from pathlib import Path

p = Path('crates/reco-detect/src/detectors/ort_gpu.rs')
t = p.read_text()
old = '''        let previous_misses = recovery.cameras[camera_index(camera)].misses;
        recovery.state_mut(camera).observe(accepted, recovery.class_id);
        recovery.stats.commits += 1;'''
new = '''        let previous_misses = recovery.cameras[camera_index(camera)].misses;
        let class_id = recovery.class_id;
        recovery.state_mut(camera).observe(accepted, class_id);
        recovery.stats.commits += 1;'''
if t.count(old) != 1:
    raise SystemExit(f'commit_ball_recovery match count={t.count(old)}')
t = t.replace(old, new, 1)
old = '''        recovery.state_mut(camera).misses = recovery
            .state_mut(camera)
            .misses
            .saturating_add(1);
        recovery.stats.rejects += 1;'''
new = '''        let state = recovery.state_mut(camera);
        state.misses = state.misses.saturating_add(1);
        recovery.stats.rejects += 1;'''
if t.count(old) != 1:
    raise SystemExit(f'reject_ball_recovery match count={t.count(old)}')
p.write_text(t.replace(old, new, 1))
print('recovery borrow fixes applied')
