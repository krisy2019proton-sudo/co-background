from kivy.app import App
from kivy.uix.button import Button
from kivy.utils import platform

class BackgroundApp(App):
    def build(self):
        return Button(
            text='Start Background Service',
            font_size=32,
            on_press=self.start_service
        )

    def start_service(self, instance):
        if platform == 'android':
            from jnius import autoclass
            # Access PythonActivity and start the service
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            activity = PythonActivity.mActivity
            Context = autoclass('android.content.Context')
            Intent = autoclass('android.content.Intent')
            
            service_intent = Intent(activity, autoclass('org.kivy.android.PythonService'))
            service_intent.putExtra('android.intent.extra.BACKGROUND_SERVICE', True)
            activity.startService(service_intent)
            instance.text = "Service Started!"

if __name__ == '__main__':
    BackgroundApp().run()
