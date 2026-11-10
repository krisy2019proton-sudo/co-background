import json
import os
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.utils import platform

CONFIG_FILE = "worker_config.json"

class WorkerConfigPanel(BoxLayout):
    def __init__(self, **kwargs):
        super(WorkerConfigPanel, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = 30
        self.spacing = 15

        config = self.load_config()

        self.add_widget(Label(text="PC IP Address:", font_size=18, size_hint_y=None, height=35))
        self.ip_input = TextInput(text=config.get("host", "192.168.3.133"), multiline=False, size_hint_y=None, height=45)
        self.add_widget(self.ip_input)

        self.add_widget(Label(text="Cluster Password:", font_size=18, size_hint_y=None, height=35))
        self.pass_input = TextInput(text=config.get("password", "rycbar"), password=True, multiline=False, size_hint_y=None, height=45)
        self.add_widget(self.pass_input)

        self.start_btn = Button(text='Start Background Worker', font_size=20, size_hint_y=None, height=55)
        self.start_btn.bind(on_press=self.start_service)
        self.add_widget(self.start_btn)
        
        self.status_label = Label(text="Status: Stopped", font_size=16)
        self.add_widget(self.status_label)

    def load_config(self):
        path = CONFIG_FILE
        if platform == 'android':
            from jnius import autoclass
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            activity = PythonActivity.mActivity
            path = os.path.join(activity.getFilesDir().getAbsolutePath(), CONFIG_FILE)
            
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}

    def save_config(self, host, password):
        path = CONFIG_FILE
        if platform == 'android':
            from jnius import autoclass
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            activity = PythonActivity.mActivity
            path = os.path.join(activity.getFilesDir().getAbsolutePath(), CONFIG_FILE)
            
        with open(path, 'w') as f:
            json.dump({"host": host, "password": password}, f)

    def start_service(self, instance):
        host = self.ip_input.text.strip()
        password = self.pass_input.text.strip()
        self.save_config(host, password)

        if platform == 'android':
            from jnius import autoclass
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            activity = PythonActivity.mActivity
            Intent = autoclass('android.content.Intent')
            
            service_intent = Intent(activity, autoclass('org.kivy.android.PythonService'))
            service_intent.putExtra('android.intent.extra.BACKGROUND_SERVICE', True)
            activity.startService(service_intent)
            
        self.status_label.text = f"Status: Running (Target: {host})"
        self.start_btn.text = "Worker Running"
        self.start_btn.disabled = True

class PhoneWorkerApp(App):
    def build(self):
        return WorkerConfigPanel()

if __name__ == '__main__':
    PhoneWorkerApp().run()
        
