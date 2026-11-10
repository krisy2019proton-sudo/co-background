import json
import os
from jnius import autoclass

PythonService = autoclass('org.kivy.android.PythonService')
service = PythonService.mService
Context = autoclass('android.content.Context')
NotificationBuilder = autoclass('androidx.core.app.NotificationCompat$Builder')
NotificationChannel = autoclass('android.app.NotificationChannel')
NotificationManager = autoclass('android.app.NotificationManager')

def show_persistent_notification():
    channel_id = "phone_worker_channel"
    ns = service.getSystemService(Context.NOTIFICATION_SERVICE)
    channel = NotificationChannel(channel_id, "GRU Phone Worker", NotificationManager.IMPORTANCE_LOW)
    ns.createNotificationChannel(channel)
    
    builder = NotificationBuilder(service, channel_id)
    builder.setContentTitle("GRU Phone Worker Active")
    builder.setContentText("Processing training tasks for PC...")
    builder.setSmallIcon(service.getResources().getIdentifier("icon", "drawable", service.getPackageName()))
    builder.setOngoing(True)
    
    service.startForeground(1, builder.build())

if __name__ == '__main__':
    show_persistent_notification()
    
    # Load configuration set in the app UI
    path = "worker_config.json"
    try:
        files_dir = service.getFilesDir().getAbsolutePath()
        path = os.path.join(files_dir, "worker_config.json")
    except:
        pass
        
    if os.path.exists(path):
        with open(path, 'r') as f:
            config = json.load(f)
            import phone_worker
            phone_worker.HOST = config.get("host", phone_worker.HOST)
            phone_worker.CLUSTER_PASSWORD = config.get("password", phone_worker.CLUSTER_PASSWORD)

    # Run your worker loop from your script
    import phone_worker
    phone_worker.run_phone(phone_worker.HOST, phone_worker.CLUSTER_PORT, phone_worker.PHONE_NAME, phone_worker.PROCESSES)
