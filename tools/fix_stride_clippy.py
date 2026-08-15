from pathlib import Path

# Keep the Windows staging index available without making it an unused Linux
# parameter in default CI builds.
p = Path('crates/reco-core/src/session/detection_dispatch.rs')
t = p.read_text()
old = '''        source_index: u64,
        analysis_index: u64,
'''
new = '''        _source_index: u64,
        analysis_index: u64,
'''
if t.count(old) != 1:
    raise SystemExit(f'detection signature match count={t.count(old)}')
t = t.replace(old, new, 1)
t = t.replace('(source_index as usize * 2) % pool.n_slots()', '(_source_index as usize * 2) % pool.n_slots()', 1)
t = t.replace('(source_index as usize * 2 + 1) % pool.n_slots()', '(_source_index as usize * 2 + 1) % pool.n_slots()', 1)
p.write_text(t)

# Clippy: collapse the EOF sparse-anchor guard without changing semantics.
p = Path('crates/reco-core/src/session/run_loop.rs')
t = p.read_text()
old = '''                if self.frame_stride > 1 {
                    if let Some(anchor) = sparse_anchor.take() {
                        queue_sparse_segment(&mut pose_queue, anchor, &mut sparse_between, None);
                    }
                }
'''
new = '''                if self.frame_stride > 1
                    && let Some(anchor) = sparse_anchor.take()
                {
                    queue_sparse_segment(&mut pose_queue, anchor, &mut sparse_between, None);
                }
'''
if t.count(old) != 1:
    raise SystemExit(f'run-loop nested-if match count={t.count(old)}')
p.write_text(t.replace(old, new, 1))
print('stride lint cleanup applied')
