# -*- coding: utf-8 -*-

from xadmin.views import BaseAdminPlugin, ListAdminView
from django.template import loader
from xadmin.sites import site


# 加载excel插件
class ExcelPlugin(BaseAdminPlugin):
    import_excel = False

    def init_request(self, *args, **kwargs):
        return bool(self.import_excel)

    def block_top_toolbar(self, context, nodes):
        nodes.append(loader.render_to_string("xadmin/excel/model_list.top_toolbar.import.html", context_instance=context))

site.register_plugin(ExcelPlugin, ListAdminView)