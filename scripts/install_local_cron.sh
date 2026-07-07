#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
TREND_DATA_WORKSPACE="${TREND_DATA_WORKSPACE:-$ROOT}"
LOG_DIR="$ROOT/data/logs"
mkdir -p "$LOG_DIR"

DAILY_TIME="${DAILY_TIME:-30 7 * * *}"
WEEKLY_TIME="${WEEKLY_TIME:-15 8 * * 1}"

DAILY_CMD="cd \"$ROOT\" && TREND_DATA_WORKSPACE=\"$TREND_DATA_WORKSPACE\" \"$PYTHON_BIN\" scripts/scheduled_pipeline.py daily-scrape >> \"$LOG_DIR/cron_daily_scrape.log\" 2>&1"
WEEKLY_CMD="cd \"$ROOT\" && TREND_DATA_WORKSPACE=\"$TREND_DATA_WORKSPACE\" \"$PYTHON_BIN\" scripts/scheduled_pipeline.py weekly-publish --push >> \"$LOG_DIR/cron_weekly_publish.log\" 2>&1"

TMP_CRON="$(mktemp)"

{
  crontab -l 2>/dev/null | grep -v "Reddit-Trend-Copilot scheduled_pipeline" || true
  echo "# Reddit-Trend-Copilot scheduled_pipeline daily raw collection"
  echo "$DAILY_TIME $DAILY_CMD # Reddit-Trend-Copilot scheduled_pipeline"
  echo "# Reddit-Trend-Copilot scheduled_pipeline weekly dashboard publish"
  echo "$WEEKLY_TIME $WEEKLY_CMD # Reddit-Trend-Copilot scheduled_pipeline"
} > "$TMP_CRON"

crontab "$TMP_CRON"
rm -f "$TMP_CRON"

echo "Installed local cron jobs:"
echo "  Daily scrape:     $DAILY_TIME"
echo "  Weekly publish:   $WEEKLY_TIME"
echo "Logs:"
echo "  $LOG_DIR/cron_daily_scrape.log"
echo "  $LOG_DIR/cron_weekly_publish.log"
echo "Data workspace:"
echo "  $TREND_DATA_WORKSPACE"
