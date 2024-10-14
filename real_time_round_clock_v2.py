import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime, timedelta
import matplotlib.animation as animation
import argparse
import sys
from PIL import Image

# Function to draw a clock face with color gradient from 12 to 6 o'clock and 70% opacity
def draw_clock_face(ax, border_color='black', gradient_color=None, position=[0, 0, 0]):
    theta = np.linspace(0, 2 * np.pi, 100)
    x = np.cos(theta) + position[0]
    y = np.sin(theta) + position[1]
    z = np.zeros_like(x) + position[2]
    
    if gradient_color:
        cmap = plt.get_cmap(gradient_color)
        for i in range(len(theta)):
            normalized_value = (1 - np.cos(theta[i])) / 2
            ax.plot([position[0], x[i]], [position[1], y[i]], [position[2], z[i]], color=cmap(normalized_value), alpha=0.7, linewidth=2)
    else:
        ax.plot(x, y, z, color=border_color, linewidth=2)

    for hour in range(12):
        angle = np.pi / 6 * hour
        x_start, y_start = 0.9 * np.cos(angle) + position[0], 0.9 * np.sin(angle) + position[1]
        x_end, y_end = np.cos(angle) + position[0], np.sin(angle) + position[1]
        ax.plot([x_start, x_end], [y_start, y_end], [position[2], position[2]], color=border_color, linewidth=1)

    ax.plot([position[0], position[0]], [position[1] + 1, position[1] + 0.85], [position[2], position[2]], color='gold', linewidth=5)

def draw_clock_hands(ax, hour, minute, second, show_second_hand=True, position=[0, 0, 0]):
    hour_angle = (np.pi / 6) * (hour % 12) + (np.pi / 360) * minute
    minute_angle = (np.pi / 30) * minute
    second_angle = (np.pi / 30) * second
    
    hour_hand_x = [position[0], position[0] + 0.5 * np.cos(np.pi/2 - hour_angle)]
    hour_hand_y = [position[1], position[1] + 0.5 * np.sin(np.pi/2 - hour_angle)]
    hour_hand_z = [position[2], position[2]]
    
    minute_hand_x = [position[0], position[0] + 0.8 * np.cos(np.pi/2 - minute_angle)]
    minute_hand_y = [position[1], position[1] + 0.8 * np.sin(np.pi/2 - minute_angle)]
    minute_hand_z = [position[2], position[2]]
    
    ax.plot(hour_hand_x, hour_hand_y, hour_hand_z, color='blue', linewidth=3)
    ax.plot(minute_hand_x, minute_hand_y, minute_hand_z, color='red', linewidth=2)

    if show_second_hand:
        second_hand_x = [position[0], position[0] + 0.9 * np.cos(np.pi/2 - second_angle)]
        second_hand_y = [position[1], position[1] + 0.9 * np.sin(np.pi/2 - second_angle)]
        second_hand_z = [position[2], position[2]]
        ax.plot(second_hand_x, second_hand_y, second_hand_z, color='green', linewidth=1)

def set_background_image(fig, image_path):
    image = Image.open(image_path)
    fig.figimage(image, xo=0, yo=0, alpha=1, zorder=-1)

def update(num, ax, start_time, border_color, gradient_color, show_second_hand, text_display, position):
    ax.cla()
    
    current_time = start_time + timedelta(seconds=num)
    hour = current_time.hour
    minute = current_time.minute
    second = current_time.second
    
    draw_clock_face(ax, border_color=border_color, gradient_color=gradient_color, position=position)
    draw_clock_hands(ax, hour, minute, second, show_second_hand, position=position)

    ax.set_box_aspect([1, 1, 0.2])
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-0.5, 0.5])
    
    ax.grid(False)
    ax.set_axis_off()
    
    time_string = current_time.strftime('%H:%M:%S')
    text_display.set_text(f'Time: {time_string}')

def main():
    parser = argparse.ArgumentParser(description="3D Clock Simulation with Customizable Features")
    parser.add_argument("--start_time", type=str, default=None, help="Start time for the clock in HH:MM:SS format (24-hour).")
    parser.add_argument("--background_image", type=str, default=None, help="Path to a background image file.")
    parser.add_argument("--border_color", type=str, default='black', help="Color of the clock border.")
    parser.add_argument("--gradient_color", type=str, default=None, help="Apply color gradient to the clock face.")
    parser.add_argument("--no_second_hand", action="store_true", help="Disable the second hand on the clock.")
    parser.add_argument("--position", type=float, nargs=3, default=[0, 0, 0], help="Position of the clock face in 3D space (x, y, z).")
    parser.add_argument("--elevation", type=float, default=90, help="Elevation angle (degrees).")
    parser.add_argument("--azimuth", type=float, default=-90, help="Azimuth angle (degrees).")
    parser.add_argument("--save_video", type=str, default=None, help="Filename to save the animation as an MPEG4 video.")

    args = parser.parse_args()

    if args.start_time:
        try:
            start_time = datetime.strptime(args.start_time, "%H:%M:%S")
        except ValueError:
            print("Invalid start time format. Use HH:MM:SS.")
            sys.exit(1)
        now = datetime.now()
        start_time = start_time.replace(year=now.year, month=now.month, day=now.day)
    else:
        start_time = datetime.now()

    fig = plt.figure(figsize=(6, 6), facecolor='none')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('none')
    ax.view_init(elev=args.elevation, azim=args.azimuth)

    if args.background_image:
        try:
            set_background_image(fig, args.background_image)
        except FileNotFoundError:
            print(f"Error: Could not find image file {args.background_image}.")
            sys.exit(1)

    show_second_hand = not args.no_second_hand
    text_display = fig.text(0.75, 0.05, '', fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.7))

    if args.save_video:
        writer = animation.FFMpegWriter(fps=24, metadata=dict(artist='Me'), bitrate=1800)
        ani = animation.FuncAnimation(
            fig, 
            update, 
            fargs=(ax, start_time, args.border_color, args.gradient_color, show_second_hand, text_display, args.position), 
            interval=42,  # 1000ms / 24fps ≈ 42ms per frame
            cache_frame_data=False
        )
        ani.save(args.save_video, writer=writer)
        print(f"Video saved as {args.save_video}")
        sys.exit(0)

    ani = animation.FuncAnimation(
        fig, 
        update, 
        fargs=(ax, start_time, args.border_color, args.gradient_color, show_second_hand, text_display, args.position), 
        interval=1000,
        cache_frame_data=False
    )

    plt.show()

if __name__ == "__main__":
    main()
