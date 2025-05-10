import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
import threading
import time
import os
from groq import Groq  # Only cloud dependency (no local models)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

class VirtualPTAssistant:
    def __init__(self, root):
        self.root = root
        self.root.title("Virtual Physical Therapy Assistant")
        
        # Initialize Groq client (cloud-based, no downloads)
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY", "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"))  # Free tier, no API key needed
        
        # Webcam setup
        self.cap = cv2.VideoCapture(0)
        self.current_exercise = "squats"
        self.feedback = []
        self.chat_history = []
        
        # GUI Setup
        self.setup_gui()
        
        # Start pose estimation thread
        self.running = True
        self.thread = threading.Thread(target=self.update_frame, daemon=True)
        self.thread.start()
    
    def setup_gui(self):
        # Left Pane: Camera Feed
        self.video_frame = tk.Frame(self.root, width=640, height=480)
        self.video_frame.pack(side=tk.LEFT, padx=10, pady=10)
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()

        # Right Pane: Chat Interface
        self.chat_frame = tk.Frame(self.root, width=400)
        self.chat_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Chat Log
        self.chat_log = tk.Text(self.chat_frame, height=20, width=50)
        self.chat_log.pack(fill=tk.BOTH, expand=True)
        
        # User Input
        self.user_input = tk.Entry(self.chat_frame, width=50)
        self.user_input.pack(pady=5)
        self.user_input.bind("<Return>", lambda _: self.process_user_input())
        
        # Send Button
        self.send_button = tk.Button(
            self.chat_frame, 
            text="Send", 
            command=self.process_user_input
        )
        self.send_button.pack()
        
        # Exercise Selection
        self.exercise_var = tk.StringVar(value="squats")
        exercises = ["squats", "shoulder press", "leg raises"]
        
        for ex in exercises:
            rb = tk.Radiobutton(
                self.chat_frame,
                text=ex,
                variable=self.exercise_var,
                value=ex,
                command=self.change_exercise
            )
            rb.pack(anchor=tk.W)
    
    def change_exercise(self):
        self.current_exercise = self.exercise_var.get()
        self.feedback = [f"Switched to {self.current_exercise}"]
        self.chat_log.insert(tk.END, f"System: Now demonstrating {self.current_exercise}\n")
    
    def process_user_input(self):
        user_text = self.user_input.get()
        if not user_text.strip():
            return
            
        self.chat_log.insert(tk.END, f"You: {user_text}\n")
        self.user_input.delete(0, tk.END)
        
        # Get chatbot response (using Groq cloud)
        response = self.therapy_chatbot(user_text)
        self.chat_log.insert(tk.END, f"Assistant: {response}\n")
        self.chat_log.see(tk.END)  # Auto-scroll to bottom
    
    def therapy_chatbot(self, user_input):
        try:
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",  # Fast & free alternative: "mixtral-8x7b-32768"
                messages=[
                    {"role": "system", "content": f"You are a physical therapist assisting with {self.current_exercise}. Keep responses under 3 sentences. Current feedback: {self.feedback}"},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    def analyze_pose(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
    
            if self.current_exercise == "squats":
                self.feedback = self.check_squat_form(landmarks)
            elif self.current_exercise == "shoulder press":
                self.feedback = self.check_shoulder_press(landmarks)
            elif self.current_exercise == "leg raises":
                self.feedback = self.check_leg_raises(landmarks)
        
        return frame
    
    def check_squat_form(self, landmarks):
        feedback = []
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        # Check knee alignment
        if left_knee.x < left_hip.x or right_knee.x > right_hip.x:
            feedback.append("⚠️ Knees should align with hips")
        
        # Check depth
        if (left_hip.y - left_knee.y) < 0.2:
            feedback.append("⬇️ Go deeper into the squat")
        
        return feedback if feedback else ["✅ Good form!"]
    
    def check_shoulder_press(self, landmarks):
        # Shoulder press form logic
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        
        feedback = []
        if left_elbow.y > left_shoulder.y or right_elbow.y > right_shoulder.y:
            feedback.append("⬆️ Press higher for full extension")
        return feedback if feedback else ["✅ Good shoulder press form!"]
    
    def check_leg_raises(self, landmarks):
        # Leg raise form logic
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        
        feedback = []
        if (left_knee.y - left_hip.y) < 0.3 or (right_knee.y - right_hip.y) < 0.3:
            feedback.append("⬆️ Lift legs higher for full range")
        return feedback if feedback else ["✅ Good leg raise form!"]
    
    def update_frame(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Analyze pose and resize frame
            frame = cv2.resize(frame, (640, 480))
            frame = self.analyze_pose(frame)
            
            # Display feedback on frame
            for i, line in enumerate(self.feedback):
                cv2.putText(
                    frame, line, (10, 30 + i*30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
            
            # Convert to Tkinter format
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update GUI
            self.root.after(0, lambda: self.update_gui(imgtk))
            
            time.sleep(0.03)
        
        self.cap.release()
    
    def update_gui(self, imgtk):
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
    
    def on_closing(self):
        self.running = False
        self.thread.join()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VirtualPTAssistant(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
