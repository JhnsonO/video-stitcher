// Full-screen, yaw-linear cylindrical reprojection of two calibrated fisheye planes.
struct CameraUniforms {
    inv_model: mat4x4<f32>,
    intrinsics: vec4<f32>,
    dist: vec4<f32>,
    color_scale: vec4<f32>,
    color_offset: vec4<f32>,
    flags: vec4<u32>,
};
struct Uniforms {
    left: CameraUniforms,
    right: CameraUniforms,
    camera_position: vec4<f32>,
    projection: vec4<f32>, // yaw span, yaw centre, tan(vertical_fov/2), blend width
};
@group(0) @binding(0) var left_y: texture_2d<f32>;
@group(0) @binding(1) var left_u: texture_2d<f32>;
@group(0) @binding(2) var left_v: texture_2d<f32>;
@group(0) @binding(3) var right_y: texture_2d<f32>;
@group(0) @binding(4) var right_u: texture_2d<f32>;
@group(0) @binding(5) var right_v: texture_2d<f32>;
@group(0) @binding(6) var s_video: sampler;
@group(0) @binding(7) var<uniform> uniforms: Uniforms;
struct VsOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> VsOut {
    var positions = array<vec2<f32>, 3>(vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));
    var uvs = array<vec2<f32>, 3>(vec2(0.0,1.0), vec2(2.0,1.0), vec2(0.0,-1.0));
    var out: VsOut; out.pos=vec4(positions[vid],0.0,1.0); out.uv=uvs[vid]; return out;
}
fn rgb_to_yuv(rgb: vec3<f32>) -> vec3<f32> { return vec3(0.2126*rgb.r+0.7152*rgb.g+0.0722*rgb.b,-0.1146*rgb.r-0.3854*rgb.g+0.5*rgb.b,0.5*rgb.r-0.4542*rgb.g-0.0458*rgb.b); }
fn yuv_to_rgb(v: vec3<f32>) -> vec3<f32> { return vec3(v.x+1.5748*v.z,v.x-0.1873*v.y-0.4681*v.z,v.x+1.8556*v.y); }
fn apply_color_transfer(rgb: vec3<f32>, scale: vec3<f32>, offset: vec3<f32>) -> vec3<f32> { var yuv=rgb_to_yuv(rgb); yuv=yuv*scale+offset; return clamp(yuv_to_rgb(yuv),vec3(0.0),vec3(1.0)); }
fn sample_yuv(ytex:texture_2d<f32>, utex:texture_2d<f32>, vtex:texture_2d<f32>, uv0:vec2<f32>, flags:vec4<u32>) -> vec4<f32> {
    var uv=uv0; if flags.z==1u { uv=vec2(1.0-uv.x,1.0-uv.y); }
    if flags.y==2u { return vec4(textureSample(ytex,s_video,uv).rgb,1.0); }
    let yr=textureSample(ytex,s_video,uv).r; var ur:f32; var vr:f32;
    if flags.y==1u { let c=textureSample(utex,s_video,uv); ur=c.r; vr=c.g; } else { ur=textureSample(utex,s_video,uv).r; vr=textureSample(vtex,s_video,uv).r; }
    var y:f32; var cb:f32; var cr:f32;
    if flags.w==1u { y=yr; cb=ur-0.5; cr=vr-0.5; } else { y=(yr-16.0/255.0)*(255.0/219.0); cb=(ur-128.0/255.0)*(255.0/224.0); cr=(vr-128.0/255.0)*(255.0/224.0); }
    return vec4(clamp(vec3(y+1.5748*cr,y-0.1873*cb-0.4681*cr,y+1.8556*cb),vec3(0.0),vec3(1.0)),1.0);
}
fn camera_uv(c:CameraUniforms, origin:vec3<f32>, dir:vec3<f32>) -> vec3<f32> {
    let o=(c.inv_model*vec4(origin,1.0)).xyz; let d=(c.inv_model*vec4(dir,0.0)).xyz;
    if abs(d.z)<1e-6 { return vec3(0.0,0.0,0.0); } let t=-o.z/d.z; if t<=0.0 { return vec3(0.0,0.0,0.0); }
    let local=o+t*d;
    // quad_vertices spans x=[-.5,.5], y=[-.5/aspect,.5/aspect], with vertex UV
    // x+0.5 and 0.5-y*aspect. fs_main then applies uv*2-0.5, hence:
    let aspect=c.color_offset.w;
    let uv=vec2(2.0*(local.x+0.5)-0.5, 2.0*(0.5-local.y*aspect)-0.5);
    let x=(uv.x-c.intrinsics.z)/c.intrinsics.x; let y=(uv.y-c.intrinsics.w)/c.intrinsics.y; let r=sqrt(x*x+y*y); let theta=atan(r); let t2=theta*theta;
    let td=theta*(1.0+c.dist.x*t2+c.dist.y*t2*t2+c.dist.z*t2*t2*t2+c.dist.w*t2*t2*t2*t2);
    var scale=1.0; if r>0.0 { scale=td/r; }
    let out=vec2(c.intrinsics.x*x*scale+c.intrinsics.z,c.intrinsics.y*y*scale+c.intrinsics.w);
    if out.x<0.0||out.x>1.0||out.y<0.0||out.y>1.0 { return vec3(out,0.0); } return vec3(out,1.0);
}
@fragment fn fs_cylindrical_stereo(in:VsOut) -> @location(0) vec4<f32> {
    let yaw=uniforms.projection.y+(in.uv.x-0.5)*uniforms.projection.x;
    let pitch=atan((0.5-in.uv.y)*2.0*uniforms.projection.z);
    let cp=cos(pitch); let dir=vec3(sin(yaw)*cp,sin(pitch),-cos(yaw)*cp); let origin=uniforms.camera_position.xyz;
    let lu=camera_uv(uniforms.left,origin,dir); let ru=camera_uv(uniforms.right,origin,dir);
    if lu.z==0.0&&ru.z==0.0 { return vec4(0.0); }
    var lc=vec3(0.0); var rc=vec3(0.0);
    if lu.z>0.0 { lc=apply_color_transfer(sample_yuv(left_y,left_u,left_v,lu.xy,uniforms.left.flags).rgb,uniforms.left.color_scale.xyz,uniforms.left.color_offset.xyz); }
    if ru.z>0.0 { rc=apply_color_transfer(sample_yuv(right_y,right_u,right_v,ru.xy,uniforms.right.flags).rgb,uniforms.right.color_scale.xyz,uniforms.right.color_offset.xyz); }
    if lu.z==0.0 { return vec4(rc,1.0); } if ru.z==0.0 { return vec4(lc,1.0); }
    return vec4(mix(lc,rc,smoothstep(0.0,uniforms.projection.w,ru.x)),1.0);
}
