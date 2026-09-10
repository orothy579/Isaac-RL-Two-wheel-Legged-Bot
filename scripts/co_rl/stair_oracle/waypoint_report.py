"""Offline video-linked trace report and exportable scientific plot."""

import html
import json
import csv

from scripts.co_rl.stair_oracle.waypoint_profile import JOINTS


def write_report(out, rows, summary, profile):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = [r["t"] for r in rows]
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
    for i, name in enumerate(JOINTS):
        ax = axes[i, 0]
        ax.plot(t, [r[f"{name}_torque_requested"] for r in rows], label="PD requested", alpha=.65)
        ax.plot(t, [r[f"{name}_torque_applied"] for r in rows], label="applied", linewidth=1)
        for bound in (-17, 17):
            ax.axhline(bound, color="r", linestyle=":")
        ax.set_ylabel(name.replace("_joint", "") + "\nTorque [Nm]")
        ax.legend(loc="upper right")
        key = "q" if i < 2 else "v"
        axes[i, 1].plot(t, [r[f"{name}_{key}_target"] for r in rows], label="target")
        axes[i, 1].plot(t, [r[f"{name}_{key}"] for r in rows], label="actual")
        axes[i, 1].set_ylabel("Position [rad]" if i < 2 else "Speed [rad/s]")
        axes[i, 1].legend(loc="upper right")
    for ax in axes.flat:
        ax.grid(alpha=.25)
    axes[-1, 0].set_xlabel("Simulation time [s]")
    axes[-1, 1].set_xlabel("Simulation time [s]")
    fig.suptitle("Waypoint replay — human commands, no RL policy")
    fig.tight_layout()
    fig.savefig(out / "motor_traces.png", dpi=140)
    plt.close(fig)
    checks = []
    for i, w in enumerate(profile["waypoints"]):
        if w["t"] > rows[-1]["t"]:
            continue
        r = min(rows, key=lambda r: abs(r["t"]-w["t"]))
        start = profile["waypoints"][i-1]["t"] if i else 0.
        segment = [r for r in rows if start <= r["command_t"] <= w["t"]] or [r]
        check = {"waypoint": i, "label": w.get("label", str(i)), "target_t": w["t"], "sample_t": r["t"],
                 "base_x_error_m": w["base_reference"][0]-r["base_x"],
                 "base_z_error_m": w["base_reference"][1]-r["base_z"],
                 "pitch_error_rad": w["base_reference"][2]-r["pitch"]}
        reached = True
        for j, name in enumerate(JOINTS):
            error = w["shoulder_rad"][j]-r[f"{name}_q"] if j < 2 else w["wheel_rad_s"][j-2]-r[f"{name}_v"]
            check[f"{name}_error"] = error
            check[f"{name}_segment_peak_requested_nm"] = max(abs(r[f"{name}_torque_requested"]) for r in segment)
            check[f"{name}_segment_peak_applied_nm"] = max(abs(r[f"{name}_torque_applied"]) for r in segment)
            check[f"{name}_segment_saturated_s"] = sum(r[f"{name}_saturated"] for r in segment)*summary["physics_dt_s"]
            reached = reached and abs(error) <= (0.05 if j < 2 else 1.)
        check["joint_targets_reached_0p05rad_1rad_s"] = reached
        checks.append(check)
    if checks:
        with (out / "waypoint_checks.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(checks[0]))
            writer.writeheader()
            writer.writerows(checks)
    # Downsample display only; CSV and NPZ retain every physics step.
    data = json.dumps(rows[::max(1, len(rows)//2000)], allow_nan=False).replace("<", "\\u003c")
    content = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Waypoint replay report</title>
<style>body{font:16px system-ui;max-width:1100px;margin:24px auto;background:#131a25;color:#eef3ff}video{width:640px;max-width:100%}canvas{width:100%;background:#202b3e}select,input{font:inherit}pre{white-space:pre-wrap}a{color:#8dd4ff}img{width:100%}</style>
<h1>Waypoint 물리 재생</h1><p>@@NAME@@ — RL 모델 미사용. BC 전문가 승인 전 실험 기록.</p>
<p>점프 판정: @@SUCCESS@@ · 한계 검사: @@LIMITS@@. 상세 판정은 아래 summary를 확인하세요.</p>
<video id="video" controls src="replay.mp4"></video>
<p><input id="time" type="range" min="0" step="0.005" style="width:80%"><span id="stamp"></span></p>
<p>비디오 또는 슬라이더로 시점을 선택하세요. 그래프를 클릭하면 해당 시점으로 이동합니다.</p>
<select id="signal"></select><canvas id="plot" width="1100" height="270"></canvas><pre id="sample"></pre>
<p><a href="trace.csv">200 Hz CSV</a> · <a href="waypoint_checks.csv">waypoint별 도달 오차·구간 최대 토크</a> · <a href="transitions.npz">상태/명령/다음 상태 NPZ</a> · <a href="profile.json">입력 waypoint</a> · <a href="summary.json">판정/한계/출처</a></p>
<img src="motor_traces.png" alt="관절별 토크 및 목표/실제 추종 그래프"><details><summary>실험 상세</summary><pre>@@SUMMARY@@</pre></details>
<script>
const rows=@@DATA@@, video=document.getElementById('video'), slider=document.getElementById('time'), signal=document.getElementById('signal'), canvas=document.getElementById('plot'),ctx=canvas.getContext('2d');
const keys=Object.keys(rows[0]).filter(k=>typeof rows[0][k]==='number');
for(const k of keys){const o=document.createElement('option');o.textContent=k;signal.append(o)}
signal.value='left_shoulder_joint_torque_applied';slider.max=rows.at(-1).t;
function redraw(){const time=+slider.value,key=signal.value,ys=rows.map(r=>r[key]);let low=Math.min(...ys),high=Math.max(...ys);if(high-low<1e-6){low-=1;high+=1}const range=high-low;low-=range*.1;high+=range*.1;
ctx.clearRect(0,0,1100,270);ctx.strokeStyle='#74dcbd';ctx.beginPath();rows.forEach((r,i)=>{const x=50+r.t/slider.max*1030,y=245-(r[key]-low)/(high-low)*215;if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y)});ctx.stroke();ctx.fillStyle='#fff';ctx.fillText(high.toFixed(3),5,20);ctx.fillText(low.toFixed(3),5,245);ctx.strokeStyle='#ffbc67';ctx.beginPath();const x=50+time/slider.max*1030;ctx.moveTo(x,20);ctx.lineTo(x,245);ctx.stroke();
const row=rows.reduce((a,b)=>Math.abs(b.t-time)<Math.abs(a.t-time)?b:a);document.getElementById('stamp').textContent=time.toFixed(3)+' s';document.getElementById('sample').textContent=JSON.stringify({t:row.t,phase:row.phase,[key]:row[key],distance_m:row.distance_to_riser_m,yaw_deg:row.yaw_error_deg,top_hold_s:row.top_hold_s},null,2)}
slider.oninput=()=>{if(Number.isFinite(video.duration))video.currentTime=+slider.value;redraw()};video.ontimeupdate=()=>{slider.value=video.currentTime;redraw()};signal.onchange=redraw;
canvas.onclick=e=>{slider.value=Math.max(0,Math.min(1,(e.offsetX/canvas.clientWidth*1100-50)/1030))*slider.max;slider.oninput()};redraw();
</script></html>'''
    content = content.replace("@@NAME@@", html.escape(profile.get("name", "unnamed")))
    content = content.replace("@@SUCCESS@@", str(summary["jump_success"]))
    content = content.replace("@@LIMITS@@", str(summary["limits_passed"]))
    content = content.replace("@@SUMMARY@@", html.escape(json.dumps(summary, indent=2)))
    (out / "report.html").write_text(content.replace("@@DATA@@", data), encoding="utf-8")
