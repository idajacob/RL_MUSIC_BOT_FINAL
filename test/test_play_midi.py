import pygame
import time

# Initialiser pygame og mixer
pygame.init()
pygame.mixer.init()

# Indlæs MIDI-fil
pygame.mixer.music.load("data/output_midi/generated_by_agent_2.mid")

# Afspil
pygame.mixer.music.play()

print("▶️ Afspiller MIDI...")

# Vent til afspilning er færdig
while pygame.mixer.music.get_busy():
    time.sleep(0.1)

print("✅ Afspilning færdig.")
