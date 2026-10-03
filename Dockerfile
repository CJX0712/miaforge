FROM python:3.13-slim

WORKDIR /app

# 依赖安装（先复制锁文件以利用层缓存）
COPY requirements.lock.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements.lock.txt pytest ruff==0.16.10

# 复制源码
COPY . .

# 静态检查（版本钉死的 ruff，保证 lint 结论可复现）
RUN ruff check . && ruff format --check .

# 嗅探性自检：端到端基准 + 确定性
RUN python cli.py demo --out examples/benchmark_docker.json

ENTRYPOINT ["python", "cli.py"]
CMD ["demo"]
