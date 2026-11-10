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

# Stable Android SDK and NDK configurations
android.api = 33
android.minapi = 21
android.ndk = 25b
android.sdk = 33

# Automatically accept SDK licenses for CI/CD automation
android.accept_sdk_license = True

# Background service runner
services = worker:service.py
