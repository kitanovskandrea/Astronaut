"""Smooth local display of real course-game frames and saved Q tables.

Python executes every physics frame and every Q decision in the actual game.
The browser draws the unmodified official sprites and interpolates between
received positions; it has no game physics or policy implementation. The
server keeps one latest scene, so a slow display cannot queue old frames.
"""
import json
from pathlib import Path
import queue
import secrets
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import webbrowser

from famnit_gym.envs import DIR_ENVS

from .environment import Game, VARIANTS
from .qlearning import QLearner
from .training_job import TrainingJob


ASSET_DIRECTORY = Path(DIR_ENVS) / "astronaut" / "assets"
ASSETS = frozenset(("ground.png", "ceiling.png", "astronaut_idle.png",
                    "astronaut_thrust.png", "asteroid_1.png", "asteroid_2.png",
                    "asteroid_3.png", "oxygen.png"))


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Astronaut - autonomous Q-learning</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#0b1424;color:#eaf0fa;font:16px system-ui,sans-serif}
main{max-width:1220px;margin:auto;padding:26px}h1{font-size:32px;margin:0 0 6px}
p{color:#b6c6de;line-height:1.5;margin:0 0 20px}.top{display:flex;justify-content:space-between;gap:20px;align-items:center}
.badge{color:#7ee1c7;border:1px solid #32665c;border-radius:30px;padding:8px 14px;white-space:nowrap}
.controls{display:flex;flex-wrap:wrap;gap:9px;align-items:center;margin:18px 0}
button,input{font:inherit;color:inherit;background:#1c2d47;border:1px solid #39506f;border-radius:8px;padding:10px 15px}
button{cursor:pointer}button:hover{background:#294469}button.active{background:#326bd2;border-color:#679af3}
input{width:125px}label{display:flex;gap:9px;align-items:center;margin-left:10px;color:#c8d6e9}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}
.stat{background:#152238;border:1px solid #233653;border-radius:10px;padding:14px}
.stat span{font-size:13px;color:#a4b8d5;display:block;margin-bottom:5px}.stat strong{font-size:21px}
.learning{padding:16px;background:#152238;border:1px solid #304c70;border-radius:10px;margin:15px 0}.learning[hidden]{display:none}.learning progress{width:100%;height:16px;accent-color:#65b2ff}.learning-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:12px 0;font-size:14px;color:#c6d7ef}.learning-grid strong{display:block;font-size:20px;margin-top:5px}.explain{padding:12px 16px;border-left:3px solid #67acff;background:#152238;color:#c7daf4;font-size:14px;line-height:1.5}.screen{position:relative;background:#000;border:1px solid #354765;border-radius:12px;overflow:hidden;min-height:240px}
canvas{display:block;width:100%;aspect-ratio:3/2;object-fit:contain}#fps{position:absolute;right:10px;bottom:10px;font:12px system-ui,sans-serif;color:#c3d2e5;background:#0b1424bd;border-radius:5px;padding:4px 7px;pointer-events:none}#message{margin:15px 0;color:#c3d8f4;min-height:24px}
.foot{font-size:13px;color:#8da4c2}#stop{margin-left:auto;background:#422735;border-color:#794454}
@media(max-width:700px){main{padding:15px}.top{display:block}.badge{display:inline-block;margin:10px 0}.stats{grid-template-columns:repeat(2,1fr)}h1{font-size:25px}label{margin-left:0}}
</style></head><body><main>
<div class="top"><div><h1>Astronaut</h1><p>Learn flight, then asteroid avoidance, then safe oxygen collection.</p></div><div class="badge">Autonomous AI - group 7</div></div>
<div class="controls">
<button data-variant="fly">Flight</button><button data-variant="asteroids">+ Asteroids</button><button data-variant="oxygen">+ Oxygen</button>
<label>Episode seed <input id="seed" type="number" min="0" max="4294967295" value="__SEED__"></label>
<button id="train">Train from scratch</button><button id="again">Watch trained AI</button><button id="stop">Stop server</button></div>
<div class="explain">The AI chooses thrust or coast from what it learned. It uses its height, speed and approaching objects to avoid asteroids and collect oxygen. Training uses the official rewards: +1 for a tank, -1 for a collision, and 0 otherwise.</div>
<section class="learning" id="learning" hidden><strong id="learning-title">Learning by trial and error...</strong><progress id="progress" max="100" value="0"></progress><div class="learning-grid"><div>Q updates<strong id="updates">0</strong></div><div>Episodes tried<strong id="attempts">0</strong></div><div>Exploration<strong id="epsilon">80%</strong></div><div>Recent TD loss<strong id="loss">-</strong></div></div><span id="learning-message">Training starts with an empty Q table. The new agent will fly here when ready.</span></section>
<div class="stats"><div class="stat"><span>Status</span><strong id="phase">Starting...</strong></div>
<div class="stat"><span>Game time</span><strong id="elapsed">0.0 / 20 s</strong></div>
<div class="stat"><span>Oxygen collected</span><strong id="oxygen">0</strong></div>
<div class="stat"><span>Episode reward</span><strong id="reward">0.00</strong></div></div>
<div class="screen"><canvas id="game" width="1200" height="800" aria-label="Live Astronaut course game controlled by the AI"></canvas><span id="fps">Connecting...</span></div>
<div id="message">Loading the saved agent and game graphics...</div>
<p class="foot">The AI controls every movement. Each curriculum starts with flight; later stages inherit learned values and must pass their checks. Watch trained AI uses the selected policy. New models and logs are saved separately under runs/.</p>
</main><script>
const token=__TOKEN__;let variant=__VARIANT__,refreshing=false,stopped=false;
const labels={fly:'Flight',asteroids:'Asteroids',oxygen:'Asteroids + oxygen'};
const canvas=document.querySelector('#game'),ctx=canvas.getContext('2d'),sprites={},buffer=[];
const spriteNames=['ground.png','ceiling.png','astronaut_idle.png','astronaut_thrust.png','asteroid_1.png','asteroid_2.png','asteroid_3.png','oxygen.png'];
let generation=null,clockOffset=null,raf=null,latestScene=null,assetsReady=false,lastPaint=0;
let fpsStart=performance.now(),paintCount=0,streamCount=0,lastMetrics=performance.now();
const delayMs=50;
const assetsLoaded=Promise.all(spriteNames.map(name=>new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>{sprites[name]=image;resolve();};image.onerror=reject;image.src='/assets/'+name;})));
function beginDraw(){if(assetsReady&&raf===null)raf=requestAnimationFrame(draw);}
assetsLoaded.then(()=>{assetsReady=true;beginDraw();}).catch(()=>{document.querySelector('#message').textContent='The original game graphics could not be loaded.';});
function ingest(scene){
 if(!scene){buffer.length=0;latestScene=null;generation=null;if(raf!==null)cancelAnimationFrame(raf);raf=null;ctx.clearRect(0,0,1200,800);document.querySelector('#fps').textContent='Training';return;}
 const now=performance.now();
 if(scene.generation!==generation){generation=scene.generation;buffer.length=0;clockOffset=now-scene.timestamp;fpsStart=now;lastMetrics=now;paintCount=0;streamCount=0;}
 else clockOffset=Math.min(clockOffset,now-scene.timestamp);
 latestScene=scene;buffer.push(scene);if(buffer.length>12)buffer.splice(0,buffer.length-12);streamCount++;beginDraw();
}
function draw(now){
 raf=null;if(!assetsReady||!buffer.length)return;
 const target=now-clockOffset-delayMs;
 while(buffer.length>1&&buffer[1].timestamp<=target)buffer.shift();
 const a=buffer[0],b=buffer[1]||a;
 const ratio=a===b?0:Math.max(0,Math.min(1,(target-a.timestamp)/(b.timestamp-a.timestamp)));
 ctx.fillStyle='#000';ctx.fillRect(0,0,1200,800);
 let dx=b.landscape_x-a.landscape_x;if(dx>600)dx-=1200;if(dx< -600)dx+=1200;
 const landscape=((a.landscape_x+ratio*dx)%1200+1200)%1200;
 for(const [name,y] of [['ground.png',700],['ceiling.png',0]]){ctx.drawImage(sprites[name],landscape,y);ctx.drawImage(sprites[name],landscape-1200,y);}
 const later=new Map(b.entities.map(e=>[e.id,e]));
 for(const entity of a.entities){const next=later.get(entity.id);const x=next?entity.x+ratio*(next.x-entity.x):entity.x;const y=next?entity.y+ratio*(next.y-entity.y):entity.y;ctx.drawImage(sprites[entity.sprite],x,y,entity.width,entity.height);}
 paintCount++;
 const finished=!latestScene.playing&&target>=latestScene.timestamp,display=finished?latestScene:a;
 if(finished||now-lastPaint>100){document.querySelector('#elapsed').textContent=(display.frames/60).toFixed(1)+' / 20 s';document.querySelector('#oxygen').textContent=display.oxygen;document.querySelector('#reward').textContent=display.reward.toFixed(2);lastPaint=now;}
 if(now-lastMetrics>=500){document.querySelector('#fps').textContent=Math.round(paintCount*1000/(now-fpsStart))+' canvas fps · '+Math.round(streamCount*1000/(now-fpsStart))+' frames/s';paintCount=0;streamCount=0;fpsStart=now;lastMetrics=now;}
 // No extrapolation: after the final buffered scene, stop scheduling paints.
 if(latestScene.playing||target<latestScene.timestamp+20)raf=requestAnimationFrame(draw);
 else document.querySelector('#fps').textContent='Episode ended';
}
const events=new EventSource('/events');
events.addEventListener('scene',event=>ingest(JSON.parse(event.data)));
events.onerror=()=>{if(!stopped)document.querySelector('#fps').textContent='Reconnecting...';};
async function command(path,body={}){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json','X-Astronaut-Token':token},body:JSON.stringify(body)});if(!r.ok)throw Error(await r.text());}
async function start(v,train=false){try{await command(train?'/train':'/start',{variant:v,seed:Number(document.querySelector('#seed').value)});variant=v;document.querySelector('#message').textContent=train?'Starting a fresh Q-learning experiment...':'Starting '+labels[v]+'...';}catch(e){document.querySelector('#message').textContent=e.message;}}
document.querySelectorAll('[data-variant]').forEach(b=>b.onclick=()=>start(b.dataset.variant));
document.querySelector('#again').onclick=()=>start(variant);
document.querySelector('#train').onclick=()=>start(variant,true);
document.querySelector('#stop').onclick=async()=>{await command('/stop');stopped=true;events.close();if(raf!==null)cancelAnimationFrame(raf);raf=null;document.querySelector('#fps').textContent='Stopped';document.querySelector('#phase').textContent='Stopped';document.querySelector('#message').textContent='Replay server stopped. You can close this tab.';};
async function refresh(){if(refreshing||stopped)return;refreshing=true;try{
 const r=await fetch('/status',{cache:'no-store'});if(!r.ok)throw Error('Cannot read replay status');const s=await r.json();variant=s.variant;
 document.querySelector('#phase').textContent=s.phase;
 if(!latestScene){document.querySelector('#elapsed').textContent=(s.frames/60).toFixed(1)+' / 20 s';document.querySelector('#oxygen').textContent=s.oxygen;document.querySelector('#reward').textContent=s.reward.toFixed(2);}
 document.querySelector('#message').textContent=s.message;
 const l=s.learning;document.querySelector('#learning').hidden=!l;if(l){
 document.querySelector('#progress').value=100*l.decision/l.budget;
 document.querySelector('#updates').textContent=l.decision.toLocaleString()+' / '+l.budget.toLocaleString();
 document.querySelector('#attempts').textContent=l.episodes.toLocaleString();
 document.querySelector('#epsilon').textContent=(100*l.epsilon).toFixed(1)+'%';
 document.querySelector('#loss').textContent=l.td_mse===null?'-':l.td_mse.toFixed(3);
 const stageName=labels[l.variant]||'Flight';
 document.querySelector('#learning-title').textContent=l.phase==='complete'?'Curriculum completed - watching the learned policy':l.phase==='blocked'?'Stage not passed: '+stageName:l.phase==='validating'?'Checking '+stageName+'...':l.phase==='evaluating'?'Testing the selected policy...':l.phase==='error'?'Training could not finish':'Learning '+stageName+' by trial and error...';
 document.querySelector('#learning-message').textContent=l.phase==='complete'?'The selected policy passed its behaviour checks. Its Q table controls every movement.':l.phase==='blocked'?(l.reason||'This stage has not met its behaviour checks. Later stages will not start.'):'The AI learns flight, then avoidance, then oxygen collection. It advances only after the current stage passes its checks.';
 }
 document.querySelectorAll('[data-variant]').forEach(b=>b.classList.toggle('active',b.dataset.variant===s.variant));
 }catch(e){document.querySelector('#message').textContent='The local replay server is unavailable. Check the terminal for details.';}finally{refreshing=false;}}
refresh();setInterval(refresh,250);
</script></body></html>'''


class Replay:
    def __init__(self, model_path, seed, episodes=1, training_budget=None):
        self.model_path = Path(model_path).resolve()
        self.agent = QLearner.load(self.model_path)
        if self.agent.metadata.get('qualified') is not True:
            raise ValueError('This saved policy has not passed its curriculum stage checks.')
        self.variant = self.agent.metadata["variant"]
        self.initial_variant = self.variant
        self.model_seed = self.agent.metadata["seed"]
        self.initial_seed = seed
        self.episodes = episodes
        self.completed = 0
        self.training_budget = training_budget
        self.training_job = None
        self.learned_models = {}
        self.token = secrets.token_urlsafe(24)
        self.commands = queue.Queue()
        self.lock = threading.Lock()
        self.changed = threading.Condition(self.lock)
        self.scene = None
        self.scene_json = b"null"
        self.generation = 0
        self.object_tracks = {}
        self.next_frame_at = 0.0
        self.game = None
        self.state = None
        self.running = False
        self.stopped = False
        self.data = {"variant": self.variant, "phase": "Starting", "frames": 0,
                     "oxygen": 0, "reward": 0.0, "frame_id": 0,
                     "message": "Loading the actual course game...", "learning": None}

    def _model_for(self, variant):
        path = self.model_path
        if variant != self.initial_variant:
            path = path.parent / f"{variant}_seed{self.model_seed}.json.gz"
        return self.learned_models.get(variant, path)

    def start(self, variant, seed):
        if self.training_job is not None:
            self.training_job.stop()
            self.training_job = None
        agent = QLearner.load(self._model_for(variant))
        if agent.metadata.get('qualified') is not True:
            raise ValueError('This saved policy has not passed its curriculum stage checks.')
        if self.game is not None:
            self.game.close()
        # The actual simulation stays headless. Canvas uses the official PNG
        # assets and recorded positions, without SDL encoding or a second tick.
        self.game = Game(variant, repeat=agent.metadata["repeat"], render=False,
                         profile=agent.metadata.get("profile", "official"), gamma=agent.gamma,
                         state_spec=agent.metadata["state_spec"])
        self.state = self.game.reset(seed)
        self.agent, self.variant = agent, variant
        self.running = True
        self.generation += 1
        self.object_tracks.clear()
        self.next_frame_at = time.perf_counter() + 1 / 60
        with self.changed:
            self.data.update(variant=variant, phase="Playing", frames=0, oxygen=0,
                             reward=0.0, learning=None, message="The learned Q table is choosing every action.")
            self._publish_locked()

    def _wrapper(self, field):
        """Find render-only course data without changing its public observation."""
        wrapper = self.game.env
        while wrapper is not None:
            if field in vars(wrapper):
                return wrapper
            wrapper = vars(wrapper).get("env")
        return None

    def _objects(self, kind):
        wrapper = self._wrapper("_" + kind + "_front")
        if wrapper is None:
            return []
        front = vars(wrapper)["_" + kind + "_front"]
        back = vars(wrapper)["_" + kind + "_back"]
        track = self.object_tracks.get(kind)
        if track is None:
            track = {"front_x": float(front["x"]), "front": 1, "back": 0}
            self.object_tracks[kind] = track
        elif float(front["x"]) > track["front_x"] + 100:
            # Course wrappers copy the previous front object to their back
            # slot on respawn. Carry its id with it across that slot change.
            track["back"], track["front"] = track["front"], track["front"] + 1
        track["front_x"] = float(front["x"])
        entities = []
        for slot, value in (("front", front), ("back", back)):
            if kind == "asteroid":
                if int(value["type"]) == 0:
                    continue
                sprite = "asteroid_%s.png" % int(value["type"])
            else:
                if not value["seen"]:
                    continue
                sprite = "oxygen.png"
            entities.append({"id": "%s-%s" % (kind, track[slot]), "sprite": sprite,
                             "x": round(float(value["x"] - value["anchor_x"]), 3),
                             "y": round(float(800 - value["y"] - value["anchor_y"]), 3),
                             "width": int(value["width"]), "height": int(value["height"])})
        return entities

    def _publish_locked(self):
        """Replace, rather than enqueue, the scene. Caller holds changed lock."""
        base = self.game.env.unwrapped
        astronaut = base._astronaut
        entities = [{"id": "astronaut", "sprite": "astronaut_thrust.png" if astronaut["thrust"] else "astronaut_idle.png",
                     "x": round(float(astronaut["x"] - astronaut["anchor_x"]), 3),
                     "y": round(float(800 - astronaut["y"] - astronaut["anchor_y"]), 3),
                     "width": int(astronaut["width"]), "height": int(astronaut["height"])}]
        entities.extend(self._objects("asteroid"))
        entities.extend(self._objects("oxygen"))
        self.data["frame_id"] += 1
        self.scene = {"frame_id": self.data["frame_id"], "generation": self.generation,
                      "timestamp": round(time.perf_counter() * 1000, 3),
                      "frames": self.data["frames"], "variant": self.variant,
                      "playing": self.running, "oxygen": self.data["oxygen"],
                      "reward": self.data["reward"],
                      "landscape_x": round(float(base._landscape["x"]), 3),
                      "entities": entities}
        self.scene_json = json.dumps(self.scene, separators=(",", ":")).encode()
        self.changed.notify_all()

    def _pace(self):
        # Fixed physics cadence, without playing a backlog after a slow frame.
        delay = self.next_frame_at - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        now = time.perf_counter()
        if now - self.next_frame_at > 1 / 60:
            self.next_frame_at = now + 1 / 60
        else:
            # Absolute deadlines compensate for small OS sleep overshoots;
            # adding a fresh interval to every wakeup would drift to ~50 Hz.
            self.next_frame_at += 1 / 60

    def _on_frame(self, observation, reward, terminated, truncated, info, frame):
        self._pace()
        with self.changed:
            self.data["frames"] = int(info["steps"])
            self.data["reward"] += float(reward)
            self.data["oxygen"] = int(info.get("oxygen_count", 0))
            if terminated or truncated:
                self.running = False
                if terminated:
                    self.data.update(phase="Collision", message="Episode ended: " + info["collision"] + ". Watch another attempt or train a fresh agent.")
                else:
                    self.data.update(phase="Survived", message="The agent survived the full episode. Watch another attempt or select another game.")
            self._publish_locked()

    def start_training(self, variant, seed):
        if self.training_job is not None:
            self.training_job.stop()
        if self.game is not None:
            self.game.close()
            self.game = None
        self.running = False
        self.variant, self.initial_seed, self.completed = variant, seed, 0
        selected = QLearner.load(self._model_for(variant))
        self.training_job = TrainingJob(variant, self.model_seed, self.training_budget,
                                        profile=selected.metadata.get("profile", "official"))
        with self.changed:
            self.scene = None
            self.scene_json = b"null"
            self.data["frame_id"] += 1
            self.data.update(variant=variant, phase="Training", frames=0, oxygen=0,
                             reward=0.0, learning=self.training_job.snapshot(),
                             message="The AI is learning from repeated attempts. Its new policy will fly automatically when training finishes.")
            self.changed.notify_all()

    def poll_training(self):
        job = self.training_job
        if job is None:
            return
        progress = job.snapshot()
        with self.lock:
            self.data["learning"] = progress
            if progress["phase"] in ("training", "validating", "evaluating"):
                self.data["phase"] = {"training": "Training", "validating": "Validating checkpoint",
                                      "evaluating": "Testing selected policy"}[progress["phase"]]
        if progress["phase"] == "error":
            self.training_job = None
            with self.lock:
                self.data.update(phase="Training error", message="Training failed. Check the training.log in the latest runs/ folder.")
        elif progress["phase"] == "blocked":
            self.training_job = None
            with self.lock:
                self.data.update(phase="Stage not passed", message=progress.get('reason',
                                 'The current stage has not met its behaviour checks; later stages were not started.'))
        elif progress["phase"] == "complete" and job.process.poll() == 0:
            self.training_job = None
            self.learned_models[self.variant] = job.model_path
            self.start(self.variant, self.initial_seed)
            with self.lock:
                self.data["learning"] = progress
                self.data["message"] = "The curriculum passed its behaviour checks. The selected learned Q table controls every movement."

    def advance(self):
        # A single Q decision still holds for the model's action repeat. The
        # callback observes each real physics frame without choosing actions.
        transition = self.game.step(self.agent.greedy(self.state), on_frame=self._on_frame)
        self.state = transition.state
        if transition.terminated or transition.truncated:
            self.completed += 1
        if not self.running and self.completed < self.episodes:
            self.start(self.variant, self.initial_seed + self.completed)


def handler_for(replay):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, content, mime, cache="no-store"):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", cache)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def events(self):
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.connection.settimeout(5)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            last_seen = -1
            try:
                while True:
                    # Idle/training clients sleep on the condition; a slow
                    # client reads the latest scene, never a queue of old ones.
                    with replay.changed:
                        replay.changed.wait_for(lambda: replay.stopped or
                            replay.data["frame_id"] != last_seen, timeout=10)
                        if replay.stopped:
                            return
                        frame_id = replay.data["frame_id"]
                        content = replay.scene_json
                    if frame_id == last_seen:
                        self.wfile.write(b": keep-alive\n\n")
                    else:
                        self.wfile.write(b"id: " + str(frame_id).encode() +
                                         b"\nevent: scene\ndata: " + content + b"\n\n")
                        last_seen = frame_id
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
                return
            finally:
                # An ended event stream cannot be reused for another HTTP
                # request; browsers may close it abruptly when reloading.
                self.close_connection = True

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/":
                page = HTML.replace("__TOKEN__", json.dumps(replay.token))
                page = page.replace("__VARIANT__", json.dumps(replay.variant))
                page = page.replace("__SEED__", str(replay.initial_seed))
                self.send(200, page.encode(), "text/html; charset=utf-8")
            elif path == "/status":
                with replay.lock:
                    content = json.dumps(replay.data).encode()
                self.send(200, content, "application/json")
            elif path == "/scene":
                with replay.lock:
                    content = replay.scene_json
                self.send(200, content, "application/json")
            elif path == "/events":
                self.events()
            elif path.startswith("/assets/") and path[len("/assets/"):] in ASSETS:
                asset = ASSET_DIRECTORY / path[len("/assets/"):]
                self.send(200, asset.read_bytes(), "image/png", "public, max-age=86400")
            else:
                self.send(404, b"Not found", "text/plain")

        def do_POST(self):
            if not secrets.compare_digest(self.headers.get("X-Astronaut-Token", ""), replay.token):
                self.send(403, b"Invalid local replay token", "text/plain")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= 4096:
                    raise ValueError("Request too large")
                data = json.loads(self.rfile.read(length) or b"{}")
                if self.path in ("/start", "/train"):
                    variant, seed = data["variant"], data["seed"]
                    if variant not in VARIANTS or type(seed) is not int or not 0 <= seed < 2 ** 32:
                        raise ValueError("Choose a game and a valid integer seed")
                    if data.get("mode", "ai") != "ai":
                        raise ValueError("The AI controls every movement; player control is disabled")
                    replay.commands.put(("train" if self.path == "/train" else "start", variant, seed))
                elif self.path == "/stop":
                    replay.commands.put(("stop",))
                else:
                    self.send(404, b"Not found", "text/plain")
                    return
            except (ValueError, KeyError, TypeError) as error:
                self.send(400, str(error).encode(), "text/plain")
                return
            self.send(200, b'{"ok":true}', "application/json")
    return Handler


def serve(model_path, seed, episodes=1, port=8765, open_browser=True,
          train_first=False, training_budget=None):
    replay = Replay(model_path, seed, episodes, training_budget)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), handler_for(replay))
    except OSError:
        if port == 0:
            raise
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(replay))
    server.daemon_threads = True
    url = f"http://127.0.0.1:{server.server_port}/"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Astronaut replay: {url}", flush=True)
    print("The AI controls the game. Use Train from scratch to learn a new policy or Watch trained AI to replay it.", flush=True)
    if open_browser:
        threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
    try:
        if train_first:
            replay.start_training(replay.variant, seed)
        else:
            replay.start(replay.variant, seed)
        while not replay.stopped:
            try:
                command = replay.commands.get(block=not replay.running,
                                              timeout=0.1 if not replay.running else None)
            except queue.Empty:
                command = None
            if command:
                if command[0] == "stop":
                    with replay.changed:
                        replay.stopped = True
                        replay.changed.notify_all()
                    continue
                if command[0] == "start":
                    replay.initial_seed, replay.completed, replay.episodes = command[2], 0, 1
                    replay.start(command[1], command[2])
                if command[0] == "train":
                    replay.episodes = 1
                    replay.start_training(command[1], command[2])
            replay.poll_training()
            if replay.running:
                replay.advance()
    except KeyboardInterrupt:
        print("Replay stopped.", flush=True)
    finally:
        with replay.changed:
            replay.stopped = True
            replay.changed.notify_all()
        if replay.training_job is not None:
            replay.training_job.stop()
        if replay.game is not None:
            replay.game.close()
        server.shutdown()
        server.server_close()
