# Deploy OCR Engine to AlmaLinux 8.9

Target:

- Host: `203.194.113.161`
- Domain: `ocr.automated-underwriting.com`
- App path: `/opt/ocr-engine`
- Internal app port: `127.0.0.1:8000`

## 1. DNS

Create this DNS record before issuing HTTPS:

```text
ocr.automated-underwriting.com A 203.194.113.161
```

Verify:

```bash
dig +short ocr.automated-underwriting.com
```

## 2. Install Server Packages

```bash
dnf update -y
dnf install -y git nginx firewalld certbot python3-certbot-nginx dnf-plugins-core
dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
dnf install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
systemctl enable --now docker nginx firewalld
firewall-cmd --permanent --add-service=http
firewall-cmd --permanent --add-service=https
firewall-cmd --reload
```

## 3. Upload Project

From the workstation, after SSH access is authorized:

```powershell
scp -r "C:\Users\user\Documents\New project 2" root@203.194.113.161:/opt/ocr-engine
```

If `/opt/ocr-engine` already exists, copy with `rsync` or remove the old directory intentionally first.

## 4. Configure Environment

On VPS:

```bash
cd /opt/ocr-engine/deploy
cp ocr-engine.env.example ocr-engine.env
openssl rand -hex 32
vi ocr-engine.env
```

Set `OCR_API_KEY` to the generated value.

## 5. Start OCR Service

```bash
cd /opt/ocr-engine/deploy
docker compose up -d --build
docker compose logs -f ocr-engine
```

The first start may take longer because PaddleOCR downloads or initializes model files.

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## 6. Install Systemd Unit

```bash
cp /opt/ocr-engine/deploy/systemd/ocr-engine.service /etc/systemd/system/ocr-engine.service
systemctl daemon-reload
systemctl enable --now ocr-engine
systemctl status ocr-engine
```

## 7. Configure Nginx and HTTPS

```bash
cp /opt/ocr-engine/deploy/nginx/ocr.automated-underwriting.com.conf /etc/nginx/conf.d/ocr.automated-underwriting.com.conf
nginx -t
systemctl reload nginx
certbot --nginx -d ocr.automated-underwriting.com
```

After certbot completes:

```bash
curl https://ocr.automated-underwriting.com/health
```

## 8. Test OCR API

```bash
curl -X POST "https://ocr.automated-underwriting.com/ocr/ktp?mode=fast" \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "file=@sample-ktp.jpg"
```

STNK:

```bash
curl -X POST "https://ocr.automated-underwriting.com/ocr/stnk?mode=fast" \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "file=@sample-stnk.jpg"
```

## Notes

- Keep OCR uploads temporary. The app uses container temp directories and `/app/tmp` for enrichment polling.
- Do not expose port `8000` publicly; Nginx should be the only public entry point.
- Use at least 4 vCPU / 8 GB RAM for a tolerable CPU-only PaddleOCR POC.
