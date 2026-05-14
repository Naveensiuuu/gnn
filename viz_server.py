import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import torch, numpy as np

from dataset import build_farm_graph, partition_by_district, DISTRICT_BOUNDS, FEATURE_DIM
from models  import build_model
from train_centralized import train_model, DEFAULT_CONFIG

app = FastAPI(title="AgriConnect GNN Dashboard")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DEVICE = torch.device("cpu")
RISK_THRESHOLD = 0.65
MEDIUM_THRESHOLD = 0.35
FEATURE_NAMES = [
    "crop_type","soil_type","growth_stage","agro_zone","farm_size",
    "temperature","humidity","rainfall_7d","disease_detected","cnn_confidence",
    "outbreak_history","district_Kolar","district_Tumkur","district_Hassan",
    "district_Mandya","district_Mysuru","lat_normalised",
]

_s = {"data":None,"farms":None,"models":{},"results":{},"probs":{}}

def get_risk_tier(p):
    if p >= RISK_THRESHOLD: return "HIGH"
    if p >= MEDIUM_THRESHOLD: return "MEDIUM"
    return "LOW"

@app.on_event("startup")
async def startup():
    print("\n=== AgriConnect GNN Dashboard — Starting ===")
    data, farms = build_farm_graph(n_farms=300, seed=42, save_meta=False)
    _s["data"] = data.to(DEVICE)
    _s["farms"] = farms
    config = {**DEFAULT_CONFIG, "epochs": 80, "patience": 12}
    for name in ["graphsage","gcn","gat"]:
        print(f"\nTraining {name.upper()}...")
        result = train_model(name, config, data, save_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),"checkpoints"))
        ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)),"checkpoints",f"{name}_best.pt")
        model = build_model(name, in_channels=FEATURE_DIM).to(DEVICE)
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()
        _s["models"][name] = model
        _s["results"][name] = result["test_metrics"]
        with torch.no_grad():
            _s["probs"][name] = model(data.x, data.edge_index).cpu().numpy().tolist()
    print("\n✅ Dashboard ready at http://localhost:8000\n")

@app.get("/api/farms")
async def get_farms():
    farms = _s["farms"]; probs = _s["probs"].get("graphsage",[0.5]*len(farms))
    return JSONResponse([{
        "id":f["farm_id"],"lat":f["lat"],"lon":f["lon"],"district":f["district"],
        "crop":f["crop"],"soil":f["soil"],"stage":f["stage"],"temp":f["temperature"],
        "humidity":f["humidity"],"rain":f["rainfall_7d"],"disease":f["disease_detected"],
        "cnn_conf":round(f["cnn_confidence"],3),"history":f["outbreak_history"],"label":f["label"],
        "risk_prob":round(float(probs[i]),4),"risk_pct":int(float(probs[i])*100),
        "risk_tier":get_risk_tier(float(probs[i])),"flagged":float(probs[i])>=RISK_THRESHOLD
    } for i,f in enumerate(farms)])

@app.get("/api/benchmark")
async def get_benchmark():
    return JSONResponse(_s["results"])

@app.get("/api/districts")
async def get_districts():
    farms=_s["farms"]; probs=_s["probs"].get("graphsage",[0.5]*len(farms))
    dd={}
    for i,f in enumerate(farms):
        d=f["district"]; p=float(probs[i])
        if d not in dd: dd[d]={"total":0,"high":0,"medium":0,"low":0,"ps":[]}
        dd[d]["total"]+=1; dd[d]["ps"].append(p); dd[d][get_risk_tier(p).lower()]+=1
    return JSONResponse({d:{"total":v["total"],"high":v["high"],"medium":v["medium"],"low":v["low"],
        "avg_risk":round(float(np.mean(v["ps"]))*100,1)} for d,v in dd.items()})

@app.get("/api/shap/{farm_id}")
async def get_shap(farm_id:int):
    farms=_s["farms"]; data=_s["data"]; model=_s["models"].get("graphsage")
    if model is None or farm_id>=len(farms): return JSONResponse({"error":"not ready"})
    baseline=float(_s["probs"]["graphsage"][farm_id])
    contribs=[]
    x=data.x.clone()
    for fi in range(FEATURE_DIM):
        xp=x.clone(); delta=max(abs(float(x[farm_id,fi]))*0.15,0.05)
        xp[farm_id,fi]+=delta
        with torch.no_grad():
            pnew=model(xp,data.edge_index)[farm_id].item()
        contribs.append({"feature":FEATURE_NAMES[fi],"value":round(float(x[farm_id,fi]),3),"impact":round((pnew-baseline)/delta,4)})
    contribs.sort(key=lambda c:abs(c["impact"]),reverse=True)
    return JSONResponse({"farm_id":farm_id,"baseline":round(baseline,4),"features":contribs[:8]})

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),"dashboard.html")
    with open(html_path,"r") as f: return f.read()

if __name__=="__main__":
    import uvicorn
    uvicorn.run("viz_server:app",host="0.0.0.0",port=8000,reload=False)
