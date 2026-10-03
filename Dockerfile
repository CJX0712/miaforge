FROM python:3.13-slim

WORKDIR /app

# 依赖安装（先复制锁文件以利用层缓存）
COPY requirements.lock.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements.lock.txt pytest ruff

# 复制源码
COPY . .

# 嗅探性自检：端到端基准 + 确定性
RUN python cli.py demo --out examples/benchmark_docker.json

ENTRYPOINT ["python", "cli.py"]
CMD ["demo"]
