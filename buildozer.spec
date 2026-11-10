[app]
title = GRU Phone Worker
package.name = gruphoneworker
package.domain = org.worker
source.include_exts = py,png,jpg,kv,atlas,json
source.dir = .
version = 1.0
requirements = python3,kivy,pyjnius
orientation = portrait
fullscreen = 0
android.permissions = FOREGROUND_SERVICE, POST_NOTIFICATIONS, WAKE_LOCK, INTERNET, ACCESS_WIFI_STATE, ACCESS_NETWORK_STATE
android.api = 33
android.minapi = 21

# Automatically accept Android SDK licenses for CI/CD automation
android.accept_sdk_license = True

# Include both your worker script and the background service runner
services = worker:service.py
