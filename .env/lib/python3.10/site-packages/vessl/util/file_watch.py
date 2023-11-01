import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from vessl.util.constant import JWT_PATH

class WatchJWTHandler(FileSystemEventHandler):
    def __init__(self, logger=None, callback=None):
        self.logger = logger
        self.callback = callback

    def on_moved(self, event):
        super(WatchJWTHandler, self).on_moved(event)
        self.logger.debug(f"Moved file: from {event.src_path} to {event.dest_path}")
        if event.dest_path == JWT_PATH:
            with open(JWT_PATH, "r") as f:
                result = f.read().strip()
                self.callback(result)
            self.logger.debug(f"Moved file: {event.dest_path}")

    def on_created(self, event):
        super(WatchJWTHandler, self).on_created(event)
        if event.src_path == JWT_PATH:
            with open(JWT_PATH, "r") as f:
                result = f.read().strip()
                self.callback(result)
            self.logger.debug(f"Created file: {event.src_path}")

    def on_deleted(self, event):
        super(WatchJWTHandler, self).on_deleted(event)
        if event.src_path == JWT_PATH:
            self.logger.warning(f"Deleted file: {event.src_path}")

    def on_modified(self, event):
        super(WatchJWTHandler, self).on_modified(event)
        if event.src_path == JWT_PATH:
            self.logger.warning(f"Deleted file: {event.src_path}")

class WatchJWT:
    def __init__(self, path:str, logger, callback_function):
        self.path = path
        self.observer = Observer()
        self._token = None
        self.callback_function = callback_function
        self.logger = logger

    def run(self):
        event_handler = WatchJWTHandler(logger=self.logger, callback=self.callback_function)
        self.observer.schedule(event_handler, self.path, recursive=True)
        self.observer.start()
        try:
            while self.observer.is_alive():
                self.observer.join(1)
                
        finally:
            self.observer.stop()
            self.observer.join()         