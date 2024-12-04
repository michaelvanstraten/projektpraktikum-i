# pylint: skip-file

from manim import *


class FiniteDifferenceScene(MovingCameraScene):
    add_axis_labels = True

    def construct(self):
        # Initial scene setup
        ax = self.get_axes()

        sinc = ax.plot(
            lambda x: np.sinc(x / PI),
            color=BLUE,
            discontinuities=[0],
            use_smoothing=True,
        )
        sinc.set_stroke(width=3)
        points = sinc.get_anchors()
        right_sinc = VMobject().set_points_smoothly(points[len(points) // 2 :])
        left_sinc = (
            VMobject().set_points_smoothly(points[: len(points) // 2]).reverse_points()
        )
        VGroup(left_sinc, right_sinc).match_style(sinc).make_jagged()

        sinc_label = MathTex(r"{\sin(x) \over x}")
        sinc_label.to_corner(UL, buff=0.5)

        self.add(ax)
        self.play(
            Write(sinc_label),
            Create(right_sinc, remover=True, run_time=3),
            Create(left_sinc, remover=True, run_time=3),
        )
        self.add(sinc)

        x, dx = PI / 2, 0.75
        x_label = MathTex(r"x = \frac{\pi}{2}")
        x_label.to_corner((UP + LEFT) * 0.5)
        self.play(Write(x_label))

        slopes = ax.get_secant_slope_group(
            x=x,
            graph=sinc,
            dx=dx,
            dx_label=Tex("dx = 1.0"),
            dy_label="dy",
            dx_line_color=GREEN_B,
            secant_line_length=2,
            secant_line_color=RED_D,
        )

        # self.camera.frame.save_state() #saving camera state so that we can restore it later
        # self.play(self.camera.frame.animate.set(width = slopes.width*3).move_to(slopes))
        # self.wait()
        self.play(Create(slopes))
        self.wait()
        # self.play(Restore(self.camera.frame))

    def get_axes(
        self,
        x_range=(-10 * PI, 10 * PI, PI),
        y_range=(-0.5, 1, 0.5),
        x_length=None,
        y_length=None,
    ):
        axes = Axes(
            x_range,
            y_range,
            x_length or 1.3 * self.camera.frame_width,
            y_length or 3.5,
            tips=False,
        )
        axes.center()
        if self.add_axis_labels:
            axes.x_axis.add_labels(
                {
                    number: Tex(f"${number/PI:.0f}\\pi$")
                    for number in np.arange(*x_range)
                    if number != 0
                },
                font_size=20,
            )
            axes.y_axis.add_numbers(num_decimal_places=1, font_size=20)
        return axes
