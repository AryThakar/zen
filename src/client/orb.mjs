// Glass shading adapted from the user-supplied orb comparison. Keep its smooth
// surface, translucent interior and thin-film palette; animate real voice energy.
const noise = `
vec3 mod289v3(vec3 x){return x-floor(x*(1./289.))*289.;}
vec4 mod289v4(vec4 x){return x-floor(x*(1./289.))*289.;}
vec4 perm(vec4 x){return mod289v4(((x*34.)+1.)*x);}
vec4 tiSqrt(vec4 r){return 1.79284291400159-0.85373472095314*r;}
float snoise(vec3 v){
  const vec2 C=vec2(1./6.,1./3.);const vec4 D=vec4(0.,.5,1.,2.);
  vec3 i=floor(v+dot(v,C.yyy));vec3 x0=v-i+dot(i,C.xxx);
  vec3 g=step(x0.yzx,x0.xyz);vec3 l=1.-g;vec3 i1=min(g.xyz,l.zxy);vec3 i2=max(g.xyz,l.zxy);
  vec3 x1=x0-i1+C.xxx;vec3 x2=x0-i2+C.yyy;vec3 x3=x0-D.yyy;
  i=mod289v3(i);
  vec4 p=perm(perm(perm(i.z+vec4(0.,i1.z,i2.z,1.))+i.y+vec4(0.,i1.y,i2.y,1.))+i.x+vec4(0.,i1.x,i2.x,1.));
  float n_=0.142857142857;vec3 ns=n_*D.wyz-D.xzx;
  vec4 j=p-49.*floor(p*ns.z*ns.z);vec4 x_=floor(j*ns.z);vec4 y_=floor(j-7.*x_);
  vec4 x=x_*ns.x+ns.yyyy;vec4 y=y_*ns.x+ns.yyyy;vec4 h=1.-abs(x)-abs(y);
  vec4 b0=vec4(x.xy,y.xy);vec4 b1=vec4(x.zw,y.zw);
  vec4 s0=floor(b0)*2.+1.;vec4 s1=floor(b1)*2.+1.;
  vec4 sh=-step(h,vec4(0.));
  vec4 a0=b0.xzyw+s0.xzyw*sh.xxyy;vec4 a1=b1.xzyw+s1.xzyw*sh.zzww;
  vec3 p0=vec3(a0.xy,h.x);vec3 p1=vec3(a0.zw,h.y);vec3 p2=vec3(a1.xy,h.z);vec3 p3=vec3(a1.zw,h.w);
  vec4 norm=tiSqrt(vec4(dot(p0,p0),dot(p1,p1),dot(p2,p2),dot(p3,p3)));
  p0*=norm.x;p1*=norm.y;p2*=norm.z;p3*=norm.w;
  vec4 m=max(0.6-vec4(dot(x0,x0),dot(x1,x1),dot(x2,x2),dot(x3,x3)),0.);m=m*m;
  return 42.*dot(m*m,vec4(dot(p0,x0),dot(p1,x1),dot(p2,x2),dot(p3,x3)));
}`;
const vertex = `
precision highp float;
attribute vec3 position;
uniform float u_time, energy, aspect, voice, listen, mood, press;
uniform mat3 spin;
varying vec3 vN, vVP, vPos;
/// How far the surface sits from the centre along one direction on the sphere.
/// Every shaping term goes through this one function so the normal can be rebuilt
/// from the surface that is actually drawn, instead of assumed to be a sphere.
///
/// The reference orb is an undeformed sphere, which is why its glass reads as
/// glass: its normals are correct by construction. Displacing vertices while
/// shading with the original sphere normal makes the light, the Fresnel term and
/// the rim disagree with the shape, and the result looks like a soap bubble.
float radius(vec3 d) {
  // Speaking articulates the surface. Three bands at different scales and rates
  // keep syllables from reading as one uniform throb; the envelope driving
  // \`voice\` has a fast attack, so consonants actually land.
  float band1=sin(d.y*3.1+u_time*2.4)*cos(d.x*2.2-u_time*1.6);
  float band2=sin(d.x*4.6-u_time*3.1)*cos(d.z*3.8+u_time*2.2);
  float band3=sin(d.z*2.4+u_time*1.3)*sin(d.y*5.2-u_time*2.7);
  float speech=band1*.52+band2*.30+band3*.22;

  // Listening is an intake rather than a statement: rounder, slower, centred on
  // the vertical axis so the orb reads as leaning in.
  float intake=sin(d.y*1.7-u_time*1.15)*.55+.45;

  // Thinking stirs the interior without changing the outline much.
  float churn=sin(d.x*2.7+u_time*1.9)*sin(d.z*2.3-u_time*1.4)*mood*.022;
  // A slow breath underneath everything, so it is never entirely still.
  float breath=sin(u_time*.55)*.011+sin(u_time*.31+1.7)*.005;

  return 1.25*(1.+voice*.082*speech+listen*.046*intake+churn+breath
    +energy*.010-press*.055);
}

vec3 surface(vec3 d){ return d*radius(d); }

void main() {
  vec3 d=normalize(position);
  vec3 P=surface(d);

  // Rebuild the normal by differencing the surface across two tangents. Three
  // evaluations a vertex is cheap at this tessellation and it is what keeps the
  // shading honest while the orb moves.
  vec3 tangent=normalize(cross(abs(d.y)>.99?vec3(1.,0.,0.):vec3(0.,1.,0.),d));
  vec3 bitangent=cross(d,tangent);
  vec3 dU=surface(normalize(d+tangent*.015));
  vec3 dV=surface(normalize(d+bitangent*.015));
  vec3 n=normalize(cross(dU-P,dV-P));
  n*=sign(dot(n,d));

  // The interior pattern is sampled in object space, so the swirl is anchored in
  // the glass and turns with it rather than sliding across a spinning shell.
  vPos=d*1.25;
  vec3 p=spin*P;
  vN=normalize(spin*n);
  vVP=vec3(0.,0.,5.2)-p;
  float z=5.2-p.z;
  gl_Position=vec4(p.x*3.27085/aspect,p.y*3.27085,z*.5-1.,z);
}
`;
const fragment = `precision highp float;
uniform float energy, detail;

${noise}
varying vec3 vN;varying vec3 vVP;varying vec3 vPos;uniform float u_time;
vec3 iridescentColor(float t){
  t=fract(t);vec3 c=vec3(0.);
  c=mix(c,vec3(0.01,0.01,0.03),smoothstep(0.,.12,t));
  c=mix(c,vec3(0.05,0.18,0.72),smoothstep(0.08,.28,t));
  c=mix(c,vec3(0.0,0.72,0.88),smoothstep(0.22,.40,t));
  c=mix(c,vec3(0.1,0.92,0.78),smoothstep(0.36,.50,t));
  c=mix(c,vec3(0.90,0.95,0.98),smoothstep(0.46,.58,t));
  c=mix(c,vec3(0.95,0.72,0.18),smoothstep(0.54,.70,t));
  c=mix(c,vec3(0.72,0.14,0.55),smoothstep(0.66,.82,t));
  c=mix(c,vec3(0.32,0.04,0.48),smoothstep(0.78,.92,t));
  c=mix(c,vec3(0.01,0.01,0.03),smoothstep(0.88,1.,t));
  return c;
}

void main(){
  vec3 N=normalize(vN);vec3 V=normalize(vVP);float time=u_time*0.075;
  // The thin-film pattern was tuned against a sphere about 180 px across. Drawn
  // larger at the same frequency it stops reading as fine pearlescence in glass
  // and becomes broad bands of colour, so the sampling density follows the size
  // the orb is actually drawn at.
  vec3 sp=vec3(vPos.x*1.15,vPos.y*0.38,vPos.z*1.15)*detail;
  vec3 nv=vec3(snoise(sp+time),snoise(sp-time+10.),snoise(sp+time*0.5+20.));
  vec3 nv2=vec3(snoise(sp*1.8-time*0.6+5.),snoise(sp*2.1+time*0.4+15.),snoise(sp*1.5-time*0.3+25.));
  vec3 Np=normalize(N+nv*(0.55+energy*0.035)+nv2*0.15);
  float tF=1.-max(dot(V,N),0.);float fP=1.-max(dot(V,Np),0.);
  float rd=nv.y*0.18;float alpha=smoothstep(0.28+rd,0.58+rd,fP)+smoothstep(0.42,1.,fP)*0.28;
  vec3 R=reflect(-V,Np);float t=(R.x*0.5+R.y*0.6)+nv.x*0.18+time*0.48;
  vec3 color=iridescentColor(t);
  float dE=smoothstep(0.18,0.80,abs(R.y+R.x*0.65+nv.z*0.75));
  color=mix(vec3(0.01,0.01,0.02),color,dE);
  color=mix(vec3(dot(color,vec3(0.299,0.587,0.114))),color,1.45);
  float centerFocus=smoothstep(0.,0.72,1.-tF);
  vec3 inner1=iridescentColor(t*1.6+nv.y*1.3);
  vec3 inner2=iridescentColor(t*0.75-nv.x*1.6);
  vec3 inner3=iridescentColor(t*2.2+nv2.z*1.0);
  float swirl=smoothstep(0.18,0.92,snoise(sp*2.8+vec3(0.,time*0.45,-time*0.18)));
  float swirl2=smoothstep(0.1,0.85,snoise(sp*1.6-vec3(time*0.3,0.,time*0.2)));
  vec3 combinedInner=mix(mix(inner1,inner2,swirl),inner3,swirl2*0.35);
  vec3 glassBase=mix(vec3(0.93,0.96,0.99),combinedInner,swirl*0.65+0.22);
  color=mix(color,glassBase,centerFocus*0.52);
  alpha+=centerFocus*(0.22+swirl*0.38);
  float zDepth=1.-abs(vPos.z)/1.25;
  float rimFresnel=smoothstep(0.38,0.82,tF);
  float rim3D=rimFresnel*smoothstep(0.,0.72,zDepth);
  vec3 rimBase=mix(vec3(0.88,0.90,0.93),vec3(0.72,0.80,0.96),rim3D*0.3);
  vec3 keyLight=normalize(vec3(-0.35,0.82,0.55));
  float spec1=pow(max(dot(N,keyLight),0.),38.)*0.85;
  vec3 fillLight=normalize(vec3(0.55,0.25,0.72));
  float spec2=pow(max(dot(N,fillLight),0.),12.)*0.22;
  vec3 accentLight=normalize(vec3(0.60,-0.40,0.50));
  float spec3=pow(max(dot(N,accentLight),0.),8.)*0.12;
  vec3 specColor=vec3(0.98,0.99,1.)*spec1+vec3(0.96,0.88,0.70)*spec2+vec3(0.62,0.80,0.98)*spec3;
  float zShade=mix(0.88,1.,smoothstep(-1.,0.5,vPos.z/1.25));
  vec3 rimColor=rimBase*zShade+specColor;
  float contour=smoothstep(0.5,1.,tF)*max(0.5-N.y*0.5,0.)*0.08;
  rimColor-=vec3(contour);
  color=mix(color,rimColor,rim3D);
  alpha=mix(alpha,1.,rim3D);
  float bevel=smoothstep(0.91,1.,tF);
  color=mix(color,vec3(0.95,0.97,0.99),bevel*0.78);
  float sharpRim=smoothstep(0.84,0.93,tF)*(1.-smoothstep(0.92,1.,tF));
  color+=vec3(1.)*sharpRim*0.62;
  alpha=max(max(alpha,bevel*0.94),sharpRim*0.82);
  vec3 gleamDir=normalize(vec3(-0.5,0.72,0.48));
  float gleam=pow(max(dot(N,gleamDir),0.),180.)*1.6;
  color+=vec3(1.)*gleam;alpha=max(alpha,gleam);
  float outerRing=smoothstep(0.96,1.,tF)*(1.-smoothstep(0.97,1.,tF));
  color=mix(color,vec3(0.60,0.75,0.98),outerRing*0.5);
  alpha=max(alpha,outerRing*0.65);
  gl_FragColor=vec4(clamp(color,0.,1.),clamp(alpha,0.,1.));
}`;

export class LiquidOrb {
  /// `levels` reports the two voices separately. They are deliberately not merged:
  /// the orb should look different when it is listening to you and when it is
  /// talking back.
  constructor(canvas, levels = () => ({ input: 0, output: 0 })) {
    this.canvas = canvas;
    this.levels = levels;
    this.phase = "idle";
    this.energy = 0;
    this.voice = 0;
    this.listen = 0;
    this.mood = 0;
    this.time = 0;
    /// Free spin with inertia, so the orb has weight. A flick keeps turning and
    /// settles on its own; moving the pointer nearby nudges it rather than
    /// pinning it to the cursor.
    this.rotation = [0.18, 0.55, 0];
    this.velocity = [0, 0];
    this.dragging = false;
    this.lastPointer = [0, 0];
    this.press = 0;
    this.spin = new Float32Array(9);
    this.motion = matchMedia("(prefers-reduced-motion: reduce)");
    this.failed = false;
    this.onMove = (event) => {
      if (this.motion.matches) return;
      if (this.dragging) {
        this.velocity[1] += (event.clientX - this.lastPointer[0]) * 0.0011;
        this.velocity[0] -= (event.clientY - this.lastPointer[1]) * 0.0011;
        this.lastPointer = [event.clientX, event.clientY];
        return;
      }
      // Ease toward a drift that follows the pointer instead of snapping to it,
      // so passing the cursor over the orb turns it gently.
      const rect = canvas.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      const y = 1 - ((event.clientY - rect.top) / rect.height) * 2;
      this.velocity[1] += (x * 0.05 - this.velocity[1]) * 0.02;
      this.velocity[0] += (y * 0.05 - this.velocity[0]) * 0.02;
    };
    this.onDown = (event) => {
      if (this.motion.matches) return;
      this.dragging = true;
      this.lastPointer = [event.clientX, event.clientY];
      // A press dips the surface and lets it spring back, so touching the glass
      // answers.
      this.press = 1;
      try {
        canvas.setPointerCapture(event.pointerId);
      } catch {
        // Capture is a convenience; dragging still works through the window.
      }
    };
    this.onUp = () => {
      this.dragging = false;
    };
    canvas.addEventListener("pointerdown", this.onDown);
    addEventListener("pointermove", this.onMove);
    addEventListener("pointerup", this.onUp);
    addEventListener("pointercancel", this.onUp);
    canvas.addEventListener("webglcontextlost", (event) => {
      event.preventDefault();
      this.failed = true;
      canvas.parentElement.classList.remove("webgl");
    });
    canvas.addEventListener("webglcontextrestored", () => this.init());
    this.resize = new ResizeObserver(() => this.size());
    this.resize.observe(canvas);
    this.init();
    this.animate = (now) => {
      this.frame = requestAnimationFrame(this.animate);
      if (document.hidden || this.failed) return;
      // Draw on every frame the display offers. Gating to a fixed interval means
      // the gap between drawn frames alternates between one and two refreshes,
      // and that uneven pacing reads as stutter however high the average rate is.
      // Reduced motion is the one case that deliberately runs slowly.
      const interval = this.motion.matches ? 200 : this.phase === "idle" ? 50 : 1000 / 30;
      if (now - (this.last || 0) < interval) return;
      const dt = Math.min(0.1, (now - (this.last || now)) / 1000);
      this.last = now;
      if (!this.motion.matches) this.time += dt;
      this.draw(dt);
    };
    this.frame = requestAnimationFrame(this.animate);
  }

  init() {
    try {
      const gl = this.canvas.getContext("webgl", {
        alpha: true,
        antialias: true,
        depth: true,
        powerPreference: "low-power",
        premultipliedAlpha: false,
      });
      if (!gl) throw new Error("WebGL unavailable");
      this.gl = gl;
      const compile = (type, code) => {
        const shader = gl.createShader(type);
        gl.shaderSource(shader, code);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
          gl.deleteShader(shader);
          throw new Error("Orb shader unavailable");
        }
        return shader;
      };
      const vs = compile(gl.VERTEX_SHADER, vertex),
        fs = compile(gl.FRAGMENT_SHADER, fragment);
      const program = gl.createProgram();
      gl.attachShader(program, vs);
      gl.attachShader(program, fs);
      gl.linkProgram(program);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS))
        throw new Error("Orb shader unavailable");
      this.program = program;
      gl.useProgram(program);
      const points = [],
        indices = [],
        rows = 48,
        cols = 64;
      for (let y = 0; y <= rows; y++)
        for (let x = 0; x <= cols; x++) {
          const phi = (y / rows) * Math.PI,
            theta = (x / cols) * Math.PI * 2;
          points.push(
            Math.sin(phi) * Math.cos(theta),
            Math.cos(phi),
            Math.sin(phi) * Math.sin(theta),
          );
        }
      for (let y = 0; y < rows; y++)
        for (let x = 0; x < cols; x++) {
          const a = y * (cols + 1) + x,
            b = a + cols + 1;
          indices.push(a, b, a + 1, b, b + 1, a + 1);
        }
      this.positionBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(points), gl.STATIC_DRAW);
      const position = gl.getAttribLocation(program, "position");
      gl.enableVertexAttribArray(position);
      gl.vertexAttribPointer(position, 3, gl.FLOAT, false, 0, 0);
      this.indexBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer);
      gl.bufferData(
        gl.ELEMENT_ARRAY_BUFFER,
        new Uint16Array(indices),
        gl.STATIC_DRAW,
      );
      this.count = indices.length;
      this.uniforms = Object.fromEntries(
        [
          "u_time",
          "energy",
          "aspect",
          "spin",
          "press",
          "mood",
          "voice",
          "listen",
          "detail",
        ].map(
          (name) => [
            name,
            gl.getUniformLocation(program, name),
          ],
        ),
      );
      gl.enable(gl.DEPTH_TEST);
      gl.clearColor(0, 0, 0, 0);
      this.failed = false;
      this.size();
      this.draw();
      this.canvas.parentElement.classList.add("webgl");
    } catch {
      this.failed = true;
      this.canvas.parentElement.classList.remove("webgl");
    }
  }

  size() {
    if (!this.gl) return;
    const { width, height } = this.canvas.getBoundingClientRect();
    if (!width || !height) return;
    const scale = Math.min(
      devicePixelRatio || 1,
      1.75,
      680 / Math.max(width, height),
    );
    this.canvas.width = Math.round(width * scale);
    this.canvas.height = Math.round(height * scale);
    this.aspect = width / height;
    // The glass was tuned against a sphere about this wide. Keeping the pattern
    // the same size on screen, rather than the same size on the sphere, is what
    // keeps it reading as pearlescence in glass at every size the orb is drawn.
    this.detail = Math.max(1, Math.min(3, Math.min(width, height) / 180));
    this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
  }

  /// Rise quickly, fall slowly. A symmetric filter smooths speech into a single
  /// mound; this keeps the peaks that make movement read as talking.
  static follow(current, target, attack = 0.42, release = 0.075) {
    return current + (target - current) * (target > current ? attack : release);
  }

  /// Compose the current Euler rotation into the column-major matrix the vertex
  /// shader multiplies by.
  orientation() {
    const [rx, ry, rz] = this.rotation;
    const cx = Math.cos(rx),
      sx = Math.sin(rx),
      cy = Math.cos(ry),
      sy = Math.sin(ry),
      cz = Math.cos(rz),
      sz = Math.sin(rz);
    this.spin.set([
      cy * cz,
      cx * sz + sx * sy * cz,
      sx * sz - cx * sy * cz,
      -cy * sz,
      cx * cz - sx * sy * sz,
      sx * cz + cx * sy * sz,
      sy,
      -sx * cy,
      cx * cy,
    ]);
  }

  draw(dt = 1 / 60) {
    if (this.failed) return;
    const gl = this.gl,
      u = this.uniforms;
    const still = this.motion.matches;
    const busy = ["thinking", "preparing", "transcribing", "loading"].includes(
      this.phase,
    );
    const { input = 0, output = 0 } = this.levels() || {};
    const follow = LiquidOrb.follow;

    // Reduced motion keeps the orb present but quiet: colour still moves, the
    // silhouette does not.
    this.voice = still ? 0 : follow(this.voice, Math.min(1, output * 1.15));
    this.listen = still ? 0 : follow(this.listen, Math.min(1, input), 0.3, 0.06);
    this.mood = follow(this.mood, busy && !still ? 1 : 0, 0.05, 0.05);
    const target = still ? 0 : Math.max(input, output, busy ? 0.14 : 0.01);
    this.energy += (target - this.energy) * 0.16;

    // Inertia. The reference damps once per frame; scaling it by elapsed time
    // keeps the same weight at 60 Hz and at 144 Hz.
    const damping = Math.pow(0.86, dt * 60);
    if (!this.dragging || still) {
      this.velocity[0] *= damping;
      this.velocity[1] *= damping;
    }
    this.press *= Math.pow(0.04, dt);
    this.rotation[0] += this.velocity[0];
    this.rotation[1] += this.velocity[1];
    // Attention shows in the turn: it drifts faster while thinking, faster still
    // while speaking to you.
    if (!still)
      this.rotation[2] -= 0.0008 + this.mood * 0.0012 + this.voice * 0.001;
    this.orientation();
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.uniform1f(u.u_time, this.time);
    gl.uniform1f(u.energy, this.energy);
    gl.uniform1f(u.detail, this.detail || 1);
    gl.uniform1f(u.aspect, this.aspect || 1);
    gl.uniform1f(u.mood, this.mood);
    gl.uniform1f(u.voice, this.voice);
    gl.uniform1f(u.listen, this.listen);
    gl.uniformMatrix3fv(u.spin, false, this.spin);
    gl.uniform1f(u.press, still ? 0 : this.press);
    gl.drawElements(gl.TRIANGLES, this.count, gl.UNSIGNED_SHORT, 0);
  }

  dispose() {
    cancelAnimationFrame(this.frame);
    this.resize.disconnect();
    this.canvas.removeEventListener("pointerdown", this.onDown);
    removeEventListener("pointermove", this.onMove);
    removeEventListener("pointerup", this.onUp);
    removeEventListener("pointercancel", this.onUp);
    this.gl?.deleteBuffer(this.positionBuffer);
    this.gl?.deleteBuffer(this.indexBuffer);
    this.gl?.deleteProgram(this.program);
  }
}
