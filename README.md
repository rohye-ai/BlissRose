# 智能视觉分析系统

基于 [RF-DETR](https://github.com/roboflow/rf-detr) 以及Yolo构建的 Web 视觉分析平台，集成 **登录与 RBAC 权限**、**多推理实例**、**模型/数据集/设备管理**、**持续分析报警**、**图片/视频推理** 与 **模型训练** 等能力。

RF-DETR 采用 Transformer 端到端检测架构，相比传统 Anchor + NMS 方案：

- 无需 Anchor 框与 NMS 后处理
- 全局注意力，小目标与遮挡场景更稳健
- 更少超参，训练结果更可复现

## 功能概览

| 模块 | 功能 |
|------|------|
| 登录与 RBAC | JWT 登录、用户/角色/菜单权限、按角色动态展示侧栏 |
| 总览 | GPU 资源监控、推理实例快照、最近检测结果 |
| 模型管理 | 上传 YOLO / RF-DETR 权重，查看模型血缘 |
| 数据集 | ZIP 上传（COCO / YOLO），分页浏览图片与标注 |
| 设备管理 | 维护视频流 / 图片流设备，配置 ROI 与轮询间隔 |
| 推理实例 | 多实例并行，绑定模型、GPU、设备，启停与预热 |
| 持续分析 | 实例绑定设备后自动轮询分析，命中目标写入报警 |
| 报警管理 | 分页查看报警记录与检测快照 |
| 图片推理 | 本地上传 / URL 推理，结果可视化 |
| 视频流 | RTSP / HTTP / 摄像头，MJPEG 实时展示（手动调试） |
| 模型训练 | 选择模型与数据集创建训练任务，支持部署为新模型 |
| 全局默认 | 置信度、分辨率、视频源等默认参数 |
| 外部 API | REST 接口 + WebSocket 实时事件（需 Token） |

## 技术栈

- **后端**：FastAPI、SQLAlchemy、SQLite、JWT、bcrypt
- **推理**：RF-DETR、PyTorch、OpenCV、Supervision
- **前端**：原生 HTML / CSS / JavaScript（Ant Design Pro 风格管理界面）

## 环境要求

- Python **3.10 ~ 3.12**
- 推荐 NVIDIA GPU（8GB+ VRAM 用于训练；推理可 CPU/GPU）
- Windows / Linux
- 可选：`nvidia-smi`（用于 GPU 监控）

## 快速开始

```bash
cd BlissRose

# 创建虚拟环境（推荐）
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux

pip install -r requirements.txt

# 启动服务
python run.py
```

浏览器访问：**http://localhost:8080/login**

- 内置管理员：`admin` / `admin123`（首次启动自动创建）
- 登录后进入主界面：**http://localhost:8080**
- Swagger API 文档：**http://localhost:8080/docs**

> 生产环境请修改默认密码，并设置环境变量 `RFDETR_SECRET_KEY` 作为 JWT 签名密钥。

## 内置角色

| 角色 | 说明 |
|------|------|
| `admin` | 系统管理员，全部菜单 |
| `operator` | 操作员，推理/训练/设备/报警等业务菜单 |
| `viewer` | 只读，仅总览与 API 文档 |

超级管理员可在「用户管理」「角色管理」中维护账号与菜单权限。

## 典型使用流程

### 1. 上传模型与数据集

1. 在「模型管理」上传 `.pth` / `.pt` 权重（支持 RF-DETR、YOLO）
2. 在「数据集」上传 ZIP 包（COCO 或 YOLO 目录结构）
3. 可在数据集详情中分页浏览图片与标注

### 2. 配置推理实例

1. 进入「推理实例」，新建实例并 **选择已上传模型**，配置 GPU、置信度、分辨率等
2. 点击 **启动** 加载所选模型权重
3. 可选：绑定一个或多个 **设备**（视频流 / 图片 URL）
4. 启动 **持续分析**，系统将按设备轮询检测并产生报警

### 3. 图片推理

1. 确保目标推理实例已启动
2. 在「图片推理」选择实例，上传图片或输入 URL
3. 查看检测框、类别与耗时；总览页同步展示最近结果

### 4. 视频流（手动调试）

1. 在「视频流」选择推理实例与视频源，例如：
   - `0` — 默认摄像头
   - `rtsp://user:pass@192.168.1.100/stream`
   - `http://127.0.0.1:8554/stream.mjpg`
2. 点击 **启动视频流**，页面通过 MJPEG 实时展示

### 5. 模型训练

1. 在「模型训练」选择基座模型与数据集，配置 epochs / batch / 学习率 / GPU
2. 创建并启动训练任务，查看日志
3. 训练完成后可 **部署为新模型**，再在推理实例中选用

**数据集目录示例（COCO）：**

```
datasets/smoke_fire/
├── train/
│   ├── _annotations.coco.json
│   └── images/
├── valid/
│   ├── _annotations.coco.json
│   └── images/
└── test/          # 可选
```

**数据集目录示例（YOLO）：**

```
datasets/smoke_fire/
├── data.yaml
├── train/images/
├── valid/images/
└── test/images/
```

## 外部 API 示例

除 `/api/auth/login`、`/docs` 等公开接口外，请求需在 Header 携带：

```
Authorization: Bearer <access_token>
```

**登录：**

```bash
curl -X POST http://localhost:8080/api/auth/login \
  -H "Content-Type: application/json" \
  -d "{\"username\": \"admin\", \"password\": \"admin123\"}"
```

**图片 URL 推理：**

```bash
curl -X POST http://localhost:8080/api/infer/url \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d "{\"url\": \"https://example.com/image.jpg\", \"instance_id\": \"default\", \"confidence\": 0.5}"
```

**上传图片：**

```bash
curl -X POST http://localhost:8080/api/infer/image \
  -H "Authorization: Bearer <token>" \
  -F "file=@test.jpg" \
  -F "instance_id=default"
```

**查询系统状态（含 GPU、实例列表）：**

```bash
curl http://localhost:8080/api/status \
  -H "Authorization: Bearer <token>"
```

**WebSocket 实时事件：**

```
ws://localhost:8080/ws/events?token=<access_token>
```

推送事件包括：`inference_done`、`instance_started`、`instances_updated`、`config_updated`、`train_started` 等。

## 项目结构

```
rf-detr-platform/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI 入口、推理/视频/训练 API
│   │   ├── platform_routes.py   # 模型/数据集/设备/报警/训练任务 API
│   │   ├── rbac_routes.py       # 登录与用户/角色管理
│   │   ├── model_manager.py     # 多推理实例管理
│   │   ├── analysis_worker.py   # 设备轮询持续分析
│   │   ├── training_worker.py   # 训练任务执行
│   │   ├── video_worker.py      # 视频流 MJPEG
│   │   ├── config.py            # SQLite 配置存储
│   │   └── ...
│   └── scripts/
│       └── train_rfdetr.py      # RF-DETR 训练脚本
├── frontend/                    # Web 界面（登录页 + 主控制台）
├── config/
│   ├── default.yaml             # 默认配置模板
│   └── user.yaml                # 可选：首次启动迁移用
├── data/
│   ├── platform.db              # SQLite（用户、模型、数据集、设备等）
│   ├── uploads/                 # 上传文件
│   └── results/                 # 推理结果与报警图片
├── outputs/                     # 训练输出
├── models/                      # 模型权重
├── requirements.txt
└── run.py                       # 启动入口
```

## 配置说明

运行时配置主要存储在 SQLite `app_settings` 表（键 `app_config`）。**首次启动**会自动从 `config/default.yaml`（及可选的 `config/user.yaml`）迁移；之后通过 Web 界面或 `PUT /api/config` 修改均写入数据库。

关键参数：

| 参数 | 说明 |
|------|------|
| `model.confidence` | 检测置信度阈值 |
| `model.resolution` | 推理分辨率（如 320 / 576，越低越快） |
| `model.optimize_inference` | 是否启用推理编译优化 |
| `inference_instances` | 多推理实例列表（模型、GPU、设备等） |
| `default_instance_id` | 默认推理实例 ID |
| `video.fps_limit` | 视频流帧率上限 |
| `training.batch_size` / `grad_accum_steps` | 训练 batch（总 batch 建议保持 16 左右） |
| `training.gpu_ids` | 训练使用的 GPU 编号 |

服务监听地址在 `app_settings.server` 或 `config/default.yaml` 的 `server.host` / `server.port` 中配置。

## 参考

- [平台架构设计（v2.0）](docs/ARCHITECTURE.md)
- [RF-DETR GitHub](https://github.com/roboflow/rf-detr)
- [RF-DETR vs YOLO 架构对比](https://www.exxactcorp.com/blog/deep-learning/rf-detr-vs-yolo-transformers-in-computer-vision)
