import time
from jnius import autoclass

# Bring Android classes into Python
PythonService = autoclass('org.kivy.android.PythonService')
service = PythonService.mService
Context = autoclass('android.content.Context')
NotificationBuilder = autoclass('androidx.core.app.NotificationCompat$Builder')
NotificationChannel = autoclass('android.app.NotificationChannel')
NotificationManager = autoclass('android.app.NotificationManager')

def show_persistent_notification():
    channel_id = "python_service_channel"
    
    # Create notification channel for modern Android versions
    ns = service.getSystemService(Context.NOTIFICATION_SERVICE)
    channel = NotificationChannel(channel_id, "Python Background Service", NotificationManager.IMPORTANCE_LOW)
    ns.createNotificationChannel(channel)
    
    # Build the permanent notification
    builder = NotificationBuilder(service, channel_id)
    builder.setContentTitle("Python Script Running")
    builder.setContentText("Your background service is active...")
    builder.setSmallIcon(service.getResources().getIdentifier("icon", "drawable", service.getPackageName()))
    builder.setOngoing(True) # Makes it persistent/permanent
    
    # Promote service to a foreground service with the notification
    notification = builder.build()
    service.startForeground(1, notification)

if __name__ == '__main__':
    show_persistent_notification()
    
    # Put your background loop or actual script logic here
    counter = 0
    while True:
        print(f"Background running count: {counter}")
        counter += 1
        time.sleep(10) # Wait 10 seconds between tasks
