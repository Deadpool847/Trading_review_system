# 🚀 Deployment Guide - Daily Review Machine

## Local Development

### Quick Start

```bash
# Make startup script executable
chmod +x start.sh

# Run startup wizard
./start.sh
```

### Manual Start

```bash
# Create directories
mkdir -p data/bars data/trades data/cache models reports logs

# Install dependencies
pip install -r backend/requirements.txt

# Launch Streamlit
streamlit run streamlit_app.py

# Or use CLI
python cli.py run --date 2025-01-15 --scope daily --symbol RELIANCE
```

---

## Production Deployment

### Option 1: Streamlit Cloud

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Daily Review Machine"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to https://share.streamlit.io
   - Connect GitHub repository
   - Set main file: `streamlit_app.py`
   - Add secrets (Groww API credentials) in app settings
   - Deploy!

### Option 2: Docker

**Dockerfile:**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/bars data/trades data/cache models reports logs

EXPOSE 8502

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8502", "--server.address=0.0.0.0"]
```

**Build & Run:**

```bash
# Build image
docker build -t daily-review-machine .

# Run container
docker run -p 8502:8502 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/reports:/app/reports \
  daily-review-machine
```

### Option 3: Traditional Server (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt update
sudo apt install python3.11 python3-pip -y

# Clone project
git clone <your-repo-url>
cd daily-review-machine

# Install Python dependencies
pip3 install -r backend/requirements.txt

# Create systemd service
sudo tee /etc/systemd/system/review-machine.service << EOF
[Unit]
Description=Daily Review Machine
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment="PATH=/usr/local/bin:/usr/bin"
ExecStart=/usr/bin/python3 -m streamlit run streamlit_app.py --server.port=8502
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable review-machine
sudo systemctl start review-machine

# Check status
sudo systemctl status review-machine
```

**Nginx Reverse Proxy (Optional):**

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8502;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## Environment Variables

Create `.env` file:

```bash
# Groww API (if using environment vars)
GROWW_API_KEY=your_api_key_here
GROWW_USER_ID=your_user_id

# Optional: Database for trade logs
MONGO_URL=mongodb://localhost:27017
DB_NAME=trading_reviews
```

---

## Scheduled Reviews (Cron)

Add to crontab for automated daily reviews:

```bash
# Edit crontab
crontab -e

# Add daily review at 4 PM (after market close)
0 16 * * 1-5 cd /path/to/app && python cli.py run --scope daily --symbol NIFTY50 >> logs/cron.log 2>&1

# Weekly review every Saturday 10 AM
0 10 * * 6 cd /path/to/app && python cli.py run --scope weekly --symbol NIFTY50 >> logs/cron.log 2>&1

# Monthly review on 1st of month
0 10 1 * * cd /path/to/app && python cli.py run --scope monthly --symbol NIFTY50 >> logs/cron.log 2>&1
```

---

## Security Best Practices

1. **API Credentials**
   - Never commit credentials to git
   - Use environment variables or secrets management
   - Rotate keys regularly

2. **Network Security**
   - Use HTTPS in production (Let's Encrypt)
   - Firewall rules to restrict access
   - Consider VPN for sensitive data

3. **Data Privacy**
   - Trade logs contain sensitive PnL data
   - Encrypt data at rest if required
   - Restrict file permissions: `chmod 600 data/trades/*`

---

## Monitoring

### Application Logs

```bash
# View real-time logs
tail -f logs/app.log

# Search for errors
grep ERROR logs/app.log

# Monitor CLI runs
tail -f logs/cli.log
```

### Health Check Endpoint

Streamlit doesn't have built-in health endpoint. Add to monitoring:

```bash
# Simple health check
curl -f http://localhost:8502/_stcore/health || exit 1
```

---

## Backup Strategy

```bash
#!/bin/bash
# backup.sh - Run daily via cron

DATE=$(date +%Y%m%d)
BACKUP_DIR="/backups/review-machine/$DATE"

mkdir -p $BACKUP_DIR

# Backup models and reports
tar -czf $BACKUP_DIR/models.tar.gz models/
tar -czf $BACKUP_DIR/reports.tar.gz reports/
tar -czf $BACKUP_DIR/data.tar.gz data/

# Keep only last 30 days
find /backups/review-machine -type d -mtime +30 -exec rm -rf {} +

echo "✅ Backup complete: $BACKUP_DIR"
```

---

## Troubleshooting

### Streamlit won't start

```bash
# Check port availability
lsof -i :8502

# Kill existing process
pkill -f streamlit

# Clear cache
rm -rf ~/.streamlit/
```

### GrowwAPI connection fails

```bash
# Test API directly
python -c "import growwapi; client = growwapi.Groww(); print('OK')"

# Check credentials
echo $GROWW_API_KEY
```

### Out of memory

```bash
# Check memory usage
free -h

# Reduce data scope
# Edit config.yaml to limit lookback period
```

---

## Updates & Maintenance

```bash
# Pull latest code
git pull origin main

# Update dependencies
pip install -r backend/requirements.txt --upgrade

# Restart service
sudo systemctl restart review-machine

# Or restart Streamlit
pkill -f streamlit && streamlit run streamlit_app.py &
```

---

## Support

For issues or questions:

1. Check logs: `logs/app.log`
2. Review configuration: `config/config.yaml`
3. Test CLI separately: `python cli.py run --help`
4. Open GitHub issue with logs attached

---

**Happy Trading! 📈**
