# 🚀 Deployment Guide

This guide provides step-by-step instructions for deploying the EAMCET College Predictor to various platforms.

## 📋 Prerequisites

Before deploying, ensure you have:
- A GitHub account
- Python 3.9+ installed locally (for testing)
- Git installed

## 🌐 Deploy to Render (Recommended)

### Option 1: One-Click Deploy
1. Click the "Deploy to Render" button in the README
2. Connect your GitHub account
3. Select this repository
4. Render will automatically detect the configuration and deploy

### Option 2: Manual Deploy
1. **Fork/Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/Eamcet_college_predictor.git
   cd Eamcet_college_predictor
   ```

2. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

3. **Deploy on Render**
   - Go to [render.com](https://render.com)
   - Sign up/Login with GitHub
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Configure:
     - **Name**: `eamcet-college-predictor`
     - **Environment**: `Python 3`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `gunicorn app:app`
   - Click "Create Web Service"

4. **Environment Variables** (Optional)
   - `FLASK_ENV`: `production`
   - `PYTHON_VERSION`: `3.9.18`

## 🐳 Deploy with Docker

### Local Docker Development
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build and run manually
docker build -t eamcet-predictor .
docker run -p 5001:5001 eamcet-predictor
```

### Deploy to Docker Hub
```bash
# Build image
docker build -t yourusername/eamcet-predictor .

# Push to Docker Hub
docker push yourusername/eamcet-predictor

# Run on any server
docker run -p 5001:5001 yourusername/eamcet-predictor
```

## ☁️ Deploy to Heroku

### Option 1: Heroku CLI
```bash
# Install Heroku CLI
# Download from: https://devcenter.heroku.com/articles/heroku-cli

# Login to Heroku
heroku login

# Create Heroku app
heroku create your-app-name

# Add Python buildpack
heroku buildpacks:set heroku/python

# Deploy
git push heroku main

# Open app
heroku open
```

### Option 2: Heroku Dashboard
1. Go to [heroku.com](https://heroku.com)
2. Create new app
3. Connect GitHub repository
4. Enable automatic deploys
5. Deploy manually or wait for auto-deploy

## 🐙 GitHub Pages (Static Files Only)

For static file hosting:
```bash
# Create a branch for GitHub Pages
git checkout -b gh-pages

# Copy static files to root
cp -r static/* .
cp templates/* .

# Push to GitHub
git add .
git commit -m "Add static files for GitHub Pages"
git push origin gh-pages

# Enable GitHub Pages in repository settings
```

## 🔧 Environment Configuration

### Production Environment Variables
```bash
FLASK_ENV=production
PORT=5001
PYTHON_VERSION=3.9.18
```

### Development Environment Variables
```bash
FLASK_ENV=development
PORT=5001
DEBUG=True
```

## 📊 Monitoring and Logs

### Render
- View logs in the Render dashboard
- Monitor performance metrics
- Set up alerts for downtime

### Heroku
```bash
# View logs
heroku logs --tail

# Monitor dyno usage
heroku ps
```

### Docker
```bash
# View container logs
docker logs <container_id>

# Monitor resource usage
docker stats
```

## 🔒 Security Considerations

### Environment Variables
- Never commit sensitive data to Git
- Use environment variables for secrets
- Rotate API keys regularly

### HTTPS
- Enable HTTPS on all production deployments
- Use secure headers
- Implement rate limiting

### Data Protection
- Validate all user inputs
- Sanitize data before processing
- Implement proper error handling

## 🚨 Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Find process using port
lsof -i :5001

# Kill process
kill -9 <PID>
```

#### Dependencies Issues
```bash
# Clear pip cache
pip cache purge

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### Docker Issues
```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker build --no-cache -t eamcet-predictor .
```

### Debug Mode
```bash
# Enable debug mode locally
export FLASK_ENV=development
export DEBUG=True
python app.py
```

## 📈 Performance Optimization

### Caching
- Enable Redis for production caching
- Implement CDN for static assets
- Use browser caching headers

### Database
- Consider using PostgreSQL for large datasets
- Implement connection pooling
- Optimize database queries

### Monitoring
- Set up application monitoring (New Relic, DataDog)
- Monitor response times
- Track error rates

## 🔄 Continuous Deployment

### GitHub Actions
The repository includes a GitHub Actions workflow that:
- Runs tests on every push
- Prepares for deployment
- Can trigger automatic deployments

### Render Auto-Deploy
- Enable automatic deployments in Render
- Deploy on every push to main branch
- Rollback on failed deployments

## 📞 Support

If you encounter deployment issues:
1. Check the logs for error messages
2. Verify all dependencies are installed
3. Ensure environment variables are set correctly
4. Test locally before deploying
5. Create an issue on GitHub with detailed error information

---

**Happy Deploying! 🚀**
