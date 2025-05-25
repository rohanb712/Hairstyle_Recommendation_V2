@echo off
echo Starting AI Hairstyle Recommender Development Servers...
echo.

echo Starting Backend Server...
start "Backend" cmd /k "cd backend && python app/main.py"

echo Waiting for backend to start...
timeout /t 5 /nobreak > nul

echo Starting Frontend Server...
start "Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo Both servers are starting...
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo.
echo Press any key to exit...
pause > nul 