from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class ColorbarInnerPosition(object):
    def __init__(
        self,
        orientation="vertical",
        width="5%",
        height="50%",
        location=1,
        pad=0.5,
        tick_position=None,
    ):
        """
        width, height: inch if number, percentage of parent axes if string (like '5%')
        pad: points
        location are :
        'upper right' : 1,
        'upper left' : 2,
        'lower left' : 3,
        'lower right' : 4,
        'right' : 5,
        'center left' : 6,
        'center right' : 7,
        'lower center' : 8,
        'upper center' : 9,
        'center' : 10,
        """
        self.orientation = orientation
        if orientation == "vertical":
            self.width = width
            self.height = height
            if tick_position is None:
                tick_position = "left"
        else:
            self.width = height
            self.height = width
            if tick_position is None:
                tick_position = "bottom"
        self.location = location
        self.pad = pad
        self.tick_position = tick_position

    def get_cb_axes(self, ax):
        cax = inset_axes(
            ax,
            width=self.width,
            height=self.height,
            loc=self.location,
            borderpad=self.pad,
        )
        return cax

    def post_creation(self, colorbar):
        if self.orientation == "vertical":
            if self.tick_position == "left":
                colorbar.ax.yaxis.set_ticks_position(self.tick_position)
                colorbar.ax.yaxis.set_label_position(self.tick_position)
        else:
            if self.tick_position == "top":
                colorbar.ax.xaxis.set_ticks_position(self.tick_position)
                colorbar.ax.xaxis.set_label_position(self.tick_position)

    def get_orientation(self):
        return self.orientation


class ColorbarSetting(object):

    def __init__(
        self, cb_position, ticks_locator=None, ticks_formatter=None, cmap="jet"
    ):
        self.cb_position = cb_position
        self.ticks_locator = ticks_locator
        self.ticks_formatter = ticks_formatter
        self.cmap = cmap

    def add_colorbar(self, mappable, ax):
        fig = ax.get_figure()
        cb = fig.colorbar(
            mappable,
            ticks=self.ticks_locator,
            format=self.ticks_formatter,
            orientation=self.cb_position.get_orientation(),
            cax=self.cb_position.get_cb_axes(ax),
        )
        self.cb_position.post_creation(cb)
        if not hasattr(fig, "_plotutils_colorbars"):
            fig._plotutils_colorbars = dict()
        fig._plotutils_colorbars[ax] = cb
        return cb

    def get_cmap(self):
        return self.cmap
