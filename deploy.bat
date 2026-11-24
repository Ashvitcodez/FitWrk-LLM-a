@echo off
echo ================================================
echo Deploying FitWrk Backend with Gamification
echo ================================================
echo.

echo Deploying to Google Cloud Run...
gcloud run deploy fitwrkllmapi --source . --region us-east4 --memory 2Gi --cpu 2 --timeout 300 --allow-unauthenticated

echo.
echo ================================================
echo Deployment complete!
echo.
echo Test it:
echo curl https://fitwrkllmapi-26573695653.us-east4.run.app/api/gamification/health
echo ================================================
pause
