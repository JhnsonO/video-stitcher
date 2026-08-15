from pathlib import Path

path = Path("crates/reco-cli/src/stitch.rs")
text = path.read_text()
old = '''    anyhow::ensure!(
        (1..=reco_autocam::MAX_FRAME_STRIDE).contains(&args.frame_stride),
        "--frame-stride must be between 1 and {}, got {}",
        reco_autocam::MAX_FRAME_STRIDE,
        args.frame_stride,
    );
'''
new = '''    const MAX_FRAME_STRIDE: u64 = 4;
    anyhow::ensure!(
        (1..=MAX_FRAME_STRIDE).contains(&args.frame_stride),
        "--frame-stride must be between 1 and {MAX_FRAME_STRIDE}, got {}",
        args.frame_stride,
    );
'''
if text.count(old) != 1:
    raise SystemExit(f"expected exactly one CLI stride guard, found {text.count(old)}")
path.write_text(text.replace(old, new, 1))
print("fixed optional-autocam CLI guard")
