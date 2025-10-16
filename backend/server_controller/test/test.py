[Unit]
Description=FastAPI Orchestrator (HTTPS)
After=network.target

[Service]
User=root
WorkingDirectory=/home/ubuntu/python_projects/img_gen_orchestrator/server_controller/src
ExecStart=/home/ubuntu/.local/bin/uvicorn main:app \
  --host 0.0.0.0 \
  --port 443 \
  --ssl-keyfile /etc/letsencrypt/live/genedit.mooo.com/privkey.pem \
  --ssl-certfile /etc/letsencrypt/live/genedit.mooo.com/fullchain.pem \
  --log-level info
Restart=always
RestartSec=5
Environment="PYTHONPATH=/home/ubuntu/python_projects/img_gen_orchestrator/server_controller/src"

[Install]
WantedBy=multi-user.target