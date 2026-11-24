# shellcheck shell=bash
# The one way a script in tools/ asks CelesTrak for anything. Sourced, not run.
#
# CelesTrak's usage policy (https://celestrak.org/usage-policy.php, "Updated 2026 Nov 16", read
# 2026-11-24): "M2M (machine-to-machine) software should immediately stop querying when it receives
# any non-HTTP 200 responses and report the results to a human for investigation." A 301 is one of
# them -- "re-requesting the old URL will not make the error go away" -- and a 403 is sent "when
# explicitly enforcing our usage policy", with the reason in its body. So no redirect is followed,
# and the status, the Location and the message are printed before the calling script stops. The
# test is tools/test_celestrak_client.py, against a local server.

# celestrak_get URL OUT LABEL [REQUEST_HEADER ...]
#
# One request. The body goes to OUT.part and the response headers to OUT.http.part, and
# CELESTRAK_STATUS is set: 200, or 304 when the caller sent a header (a validator). Any other answer,
# or none, removes both partial files and exits the calling script with status 1.
celestrak_get() {
  local url="$1" out="$2" label="$3"
  shift 3
  local args=() header
  for header in "$@"; do
    args+=(-H "$header")
  done
  CELESTRAK_STATUS="$(curl -sS --max-time 300 -D "$out.http.part" -o "$out.part" -w '%{http_code}' "${args[@]}" "$url")" || {
    echo "$label: no answer from CelesTrak (curl failed); stopping" >&2
    rm -f "$out.part" "$out.http.part"
    exit 1
  }
  if [[ "$CELESTRAK_STATUS" == 200 || ("$CELESTRAK_STATUS" == 304 && $# -gt 0) ]]; then
    return 0
  fi
  echo "$label: CelesTrak answered HTTP $CELESTRAK_STATUS to $url; stopping, as its usage policy asks." >&2
  local location
  location="$(celestrak_header Location "$out.http.part")"
  if [[ -n "$location" ]]; then
    echo "  Location: $location (not followed; a person decides whether the URL changes)" >&2
  fi
  echo "  Its message:" >&2
  head -c 2000 "$out.part" | tr -cd '[:print:]\n' | sed 's/^/  | /' >&2
  echo >&2
  rm -f "$out.part" "$out.http.part"
  exit 1
}

# celestrak_header NAME FILE: the value of the first NAME header in a `curl -D` dump, CR stripped.
celestrak_header() {
  sed -n "/^$1:/I{s/^[^:]*:[[:space:]]*//;s/\r\$//;p;q}" "$2"
}
