[app]
title = Python Background App
package.name = pythonbackgroundapp
package.domain = org.background
source.include_exts = py,png,jpg,kv,atlas
source.dir = .
version = 1.0
requirements = python3,kivy,pyjnius
orientation = portrait
fullscreen = 0
android.permissions = FOREGROUND_SERVICE, POST_NOTIFICATIONS, WAKE_LOCK
android.api = 33
android.minapi = 21
# Define your background service here:
services = myservice:service.py
