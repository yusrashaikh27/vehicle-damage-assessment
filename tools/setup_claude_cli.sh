#!/usr/bin/env bash
# Make `claude` work in the VS Code terminal.
#
# Run this once, from anywhere:
#
#     bash "tools/setup_claude_cli.sh"
#
# It is safe to run more than once - every step checks before it changes anything,
# and ~/.zshrc is backed up before it is touched.
#
# The script prefers finding what you already have over installing something new,
# because "command not found" on macOS is far more often a PATH problem than a
# missing program.

set -uo pipefail

BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GREEN=$'\033[32m'
YELLOW=$'\033[33m'; RESET=$'\033[0m'
say()  { printf '%s\n' "$*"; }
ok()   { printf '%s  ok %s %s\n' "$GREEN" "$RESET" "$*"; }
warn() { printf '%s warn %s %s\n' "$YELLOW" "$RESET" "$*"; }
bad()  { printf '%s fail %s %s\n' "$RED" "$RESET" "$*"; }
head_() { printf '\n%s%s%s\n' "$BOLD" "$*" "$RESET"; }

ZSHRC="$HOME/.zshrc"
MARKER="# added by vehicle-damage-assessment/tools/setup_claude_cli.sh"

# ---------------------------------------------------------------- step 1
head_ "1. Is claude already available?"

if command -v claude >/dev/null 2>&1; then
  ok "claude is already on your PATH at: $(command -v claude)"
  say "${DIM}   version: $(claude --version 2>/dev/null || echo 'could not read version')${RESET}"
  say ""
  ok "Nothing to do. Skip to 'Using it' at the bottom of this output."
  FOUND_ALREADY=1
else
  warn "claude is not on your PATH in this shell"
  FOUND_ALREADY=0
fi

# ---------------------------------------------------------------- step 2
if [ "$FOUND_ALREADY" -eq 0 ]; then
  head_ "2. Is it installed but not on PATH?"
  say "${DIM}Checking the usual locations...${RESET}"

  CANDIDATES=(
    "$HOME/.local/bin/claude"
    "$HOME/.claude/local/claude"
    "$HOME/bin/claude"
    "/opt/homebrew/bin/claude"
    "/usr/local/bin/claude"
  )
  # Wherever npm puts global binaries on this machine, whatever that turns out to be.
  if command -v npm >/dev/null 2>&1; then
    NPM_BIN="$(npm prefix -g 2>/dev/null)/bin/claude"
    [ -n "$NPM_BIN" ] && CANDIDATES+=("$NPM_BIN")
  fi

  FOUND_AT=""
  for c in "${CANDIDATES[@]}"; do
    if [ -x "$c" ]; then FOUND_AT="$c"; break; fi
    say "${DIM}   not at $c${RESET}"
  done

  if [ -n "$FOUND_AT" ]; then
    ok "found it at $FOUND_AT"
    say "   So it is installed - your shell just cannot see it. Fixing PATH."
    NEEDS_PATH_DIR="$(dirname "$FOUND_AT")"
  else
    warn "not installed anywhere I looked"
    NEEDS_PATH_DIR=""
  fi
fi

# ---------------------------------------------------------------- step 3
if [ "$FOUND_ALREADY" -eq 0 ] && [ -z "${NEEDS_PATH_DIR:-}" ]; then
  head_ "3. Installing it"

  if ! command -v node >/dev/null 2>&1; then
    bad "Node.js is not installed, and the npm install route needs it."
    say ""
    say "Install Node first, then re-run this script:"
    say "    brew install node"
    say ""
    say "If you do not have Homebrew, get it from https://brew.sh first."
    exit 1
  fi

  NODE_MAJOR="$(node -v | sed 's/^v//' | cut -d. -f1)"
  say "${DIM}   node $(node -v)${RESET}"
  if [ "$NODE_MAJOR" -lt 18 ]; then
    bad "Node $NODE_MAJOR is too old - Claude Code needs 18 or newer."
    say "    brew upgrade node    # then re-run this script"
    exit 1
  fi
  ok "Node version is fine"

  say ""
  say "Running: npm install -g @anthropic-ai/claude-code"
  say "${DIM}(no sudo - see the note at the end if this fails with EACCES)${RESET}"
  say ""

  if npm install -g @anthropic-ai/claude-code; then
    ok "install command finished"
    hash -r 2>/dev/null || true
    if command -v claude >/dev/null 2>&1; then
      ok "claude is now on your PATH at $(command -v claude)"
      NEEDS_PATH_DIR=""
    else
      NEEDS_PATH_DIR="$(npm prefix -g 2>/dev/null)/bin"
      warn "installed, but still not on PATH - fixing that next"
    fi
  else
    bad "npm install failed."
    say ""
    say "Two things this could be:"
    say ""
    say "  1. A permissions error (EACCES). See the fix at the end of this output -"
    say "     do not use sudo."
    say ""
    say "  2. The package name has changed. It is written into this script as"
    say "     ${BOLD}@anthropic-ai/claude-code${RESET}, which was correct as of early 2025 but I"
    say "     could not check it against the live docs. If npm said '404 Not Found',"
    say "     that is what happened."
    say ""
    say "     Claude Code also ships as a single self-contained binary that needs no"
    say "     Node at all, which is probably the easier route now. Get the current"
    say "     install command from the official docs at ${BOLD}docs.claude.com${RESET} (search"
    say "     'Claude Code setup'), run it, then re-run this script - it will find the"
    say "     binary wherever the installer put it and sort out your PATH."
    exit 1
  fi
fi

# ---------------------------------------------------------------- step 4
if [ -n "${NEEDS_PATH_DIR:-}" ]; then
  head_ "4. Adding it to your PATH"

  if [ -f "$ZSHRC" ] && grep -qF "$NEEDS_PATH_DIR" "$ZSHRC" 2>/dev/null; then
    ok "$ZSHRC already mentions $NEEDS_PATH_DIR - not adding it twice"
  else
    if [ -f "$ZSHRC" ]; then
      BACKUP="$ZSHRC.backup-$(date +%Y%m%d-%H%M%S)"
      cp "$ZSHRC" "$BACKUP"
      ok "backed up your existing .zshrc to $(basename "$BACKUP")"
    else
      warn "no ~/.zshrc yet - creating one"
    fi
    {
      printf '\n%s\n' "$MARKER"
      printf 'export PATH="%s:$PATH"\n' "$NEEDS_PATH_DIR"
    } >> "$ZSHRC"
    ok "added $NEEDS_PATH_DIR to your PATH in ~/.zshrc"
  fi
  export PATH="$NEEDS_PATH_DIR:$PATH"
fi

# ---------------------------------------------------------------- step 5
head_ "5. Checking it actually works"
hash -r 2>/dev/null || true

if command -v claude >/dev/null 2>&1; then
  ok "claude resolves to $(command -v claude)"
  if claude --version >/dev/null 2>&1; then
    ok "it runs: $(claude --version 2>/dev/null)"
  else
    warn "found but '--version' did not run cleanly - try running 'claude' directly"
  fi
  say ""
  say "${BOLD}Restart your VS Code terminal before trying it${RESET} (the trash-can icon on"
  say "the terminal panel, then open a new one). A terminal that was already open is"
  say "still using the old PATH."
else
  bad "still not working"
  say "Send whatever this script printed above and I can work out why."
  exit 1
fi

# ---------------------------------------------------------------- notes
cat <<'NOTES'

Using it
--------
Open your project folder in VS Code (File > Open Folder, pick
"vehicle damage assessment"), open a terminal with Ctrl+` , and type:

    claude

The VS Code terminal starts in whatever folder you opened, so Claude Code picks
that up as the project root automatically. Nothing else to configure.

If you opened a parent folder instead and want to be sure, cd first:

    cd "/Users/yusra/ALL PROJECTS OF YUSRA/vehicle damage assessment" && claude

Optional shortcut
-----------------
To jump there from any terminal, add this to ~/.zshrc:

    vda() { cd "/Users/yusra/ALL PROJECTS OF YUSRA/vehicle damage assessment" && claude "$@"; }

Then `vda` anywhere gets you Claude Code in this project.

If the install failed with EACCES / permission denied
-----------------------------------------------------
Do NOT re-run it with sudo. That leaves root-owned files in your npm directory
and causes worse problems later. Point npm at a folder you own instead:

    mkdir -p "$HOME/.npm-global"
    npm config set prefix "$HOME/.npm-global"

then re-run this script - it will find the new location on its own.

Undoing what this changed
-------------------------
The only file touched outside npm is ~/.zshrc, one exported PATH line marked with
a comment naming this script. Delete those two lines to undo it. Backups are kept
next to it as .zshrc.backup-<timestamp>.
NOTES
