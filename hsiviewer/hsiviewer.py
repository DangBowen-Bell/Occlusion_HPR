from glob import glob
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import platform
import sys
import cv2
import json
import os.path as osp
import trimesh

sys.path.append('..')
from misc import constants, utils, data_io


class HSIViewer:  
    #* supported dataset
    DATASET = ['', 'PROXQ', 'PROX']
    
    #* common material
    LIT = "defaultLit"
    LIT_TRANSPARENCY = 'defaultLitTransparency'
    UN_LIT_TRANSPARENCY = 'defaultUnlitTransparency'
    SSR = "defaultLitSSR"
    UN_LIT_LINE = 'unlitLine'
    
    #* default material
    CAMERA = UN_LIT_LINE
    BODY1 = LIT
    BODY2 = LIT
        
    #* default color
    BG_COLOR = [0.9, 1.0, 0.9, 1.0]
    CAMERA_COLOR = [0.0, 0.0, 0.0, 1.0]
    DEPTH_COLOR = [1.0, 0.0, 0.0, 1.0]
    BODY_COLOR1 = [0.2, 1.0, 0.2, 1.0]
    BODY_COLOR2 = [0.2, 0.2, 1.0, 1.0]
    
    #* default image
    IMAGE_TYPES = ['color_kpt', 'depth', 'mask']
    
    #* munu number
    MENU_EXPORT = 11
    
    def __init__(self, width, height):
        #* window 
        #* - menu
        #* - scene
        #* - setting panel
        #*   - common
        #*   - fitting 1
        #*   - fitting 2
        #* - image panel
        
        self.window = gui.Application.instance.create_window("Open3D", width, height)
        w = self.window

        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)
        self._scene.scene.show_axes(True)
        self._scene.scene.set_background(HSIViewer.BG_COLOR)
        
        #* data
        self.dataset = ''
        self.dataio = None
        self.recording = ''
        self.frame_ids = []
        self.fittings = [''] * 2
        self.stages = [0] * 2
        self.depth_o3d = o3d.geometry.PointCloud()
        self.body_o3ds = [
            o3d.geometry.TriangleMesh(),
            o3d.geometry.TriangleMesh()
        ]

        self.default_img = o3d.geometry.Image()
        
        #* matrials                
        depth_mateial = rendering.MaterialRecord()
        depth_mateial.base_color = HSIViewer.DEPTH_COLOR
        
        camera_mateial = rendering.MaterialRecord()
        camera_mateial.base_color = HSIViewer.CAMERA_COLOR
        camera_mateial.shader = HSIViewer.CAMERA
        
        body_mateial1 = rendering.MaterialRecord()
        body_mateial1.base_color = HSIViewer.BODY_COLOR1
        body_mateial1.shader = HSIViewer.BODY1
        
        body_mateial2 = rendering.MaterialRecord()
        body_mateial2.base_color = HSIViewer.BODY_COLOR2
        body_mateial2.shader = HSIViewer.BODY2
    
        self.materials = {
            'scene': rendering.MaterialRecord(),
            'camera': camera_mateial,
            'depth': depth_mateial,
            'body1': body_mateial1,
            'body2': body_mateial2
        }
         
        #* setting panel              
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))

        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        #** common
        common = gui.CollapsableVert(
            "Common", 0.25 * em, gui.Margins(em, 0, 0, 0))

        #*** select dataset
        self._dataset_combobox = gui.Combobox()
        for dataset in HSIViewer.DATASET:
            self._dataset_combobox.add_item(dataset)
        self._dataset_combobox.set_on_selection_changed(self._on_select_dataset)
        common.add_fixed(separation_height)
        common.add_child(gui.Label("Dataset"))
        common.add_child(self._dataset_combobox)
        
        #*** select recording
        self._recording_combobox = gui.Combobox()
        self._recording_combobox.set_on_selection_changed(self._on_select_recording)
        common.add_fixed(separation_height)
        common.add_child(gui.Label("Recording"))
        common.add_child(self._recording_combobox)
        
        #*** select frame
        self._frame_idx = gui.Slider(gui.Slider.INT)
        self._frame_idx.set_limits(0, 0)
        self._frame_idx.set_on_value_changed(self._on_frame_idx)
        common.add_child(gui.Label("Frame"))
        common.add_child(self._frame_idx)  
        
        self._backward_button = gui.Button("Backward")
        self._backward_button.horizontal_padding_em = 0.5
        self._backward_button.vertical_padding_em = 0
        self._backward_button.set_on_clicked(self._on_mouse_backward)

        self._forward_button = gui.Button("Forward")
        self._forward_button.horizontal_padding_em = 0.5
        self._forward_button.vertical_padding_em = 0
        self._forward_button.set_on_clicked(self._on_mouse_forward)
        
        h = gui.Horiz(0.25 * em)
        h.add_stretch()
        h.add_child(self._backward_button)
        h.add_child(self._forward_button)
        h.add_stretch()
        common.add_child(h)
        
        #*** set background color
        self._bg_color = gui.ColorEdit()
        self._bg_color.color_value = gui.Color(
            r=HSIViewer.BG_COLOR[0], g=HSIViewer.BG_COLOR[1],
            b=HSIViewer.BG_COLOR[2], a=HSIViewer.BG_COLOR[3]
        )
        self._bg_color.set_on_value_changed(self._on_bg_color)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("BG color"))
        grid.add_child(self._bg_color)
        common.add_child(grid)
        
        #*** set depth color
        self._depth_color = gui.ColorEdit()
        self._depth_color.color_value = gui.Color(
            r=HSIViewer.DEPTH_COLOR[0], g=HSIViewer.DEPTH_COLOR[1],
            b=HSIViewer.DEPTH_COLOR[2], a=HSIViewer.DEPTH_COLOR[3]
        )
        self._depth_color.set_on_value_changed(self._on_depth_color)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Depth color"))
        grid.add_child(self._depth_color)
        common.add_child(grid)
        
        #*** save current frame      
        self._save_button = gui.Button("Save")
        self._save_button.horizontal_padding_em = 0.5
        self._save_button.vertical_padding_em = 0
        self._save_button.set_on_clicked(self._on_mouse_save)
        
        h = gui.Horiz(0.25 * em)
        h.add_stretch()
        h.add_child(self._save_button)
        h.add_stretch()
        common.add_child(h)
        
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(common)
        
        #** fitting 1        
        fitting_1 = gui.CollapsableVert(
            "Fitting 1", 0.25 * em, gui.Margins(em, 0, 0, 0))
        
        self._fitting_combobox_1 = gui.Combobox()
        self._fitting_combobox_1.set_on_selection_changed(self._on_select_fitting_1)
        fitting_1.add_fixed(separation_height)
        fitting_1.add_child(self._fitting_combobox_1)
        
        self._body_color_1 = gui.ColorEdit()
        self._body_color_1.color_value = gui.Color(
            r=HSIViewer.BODY_COLOR1[0], g=HSIViewer.BODY_COLOR1[1],
            b=HSIViewer.BODY_COLOR1[2], a=HSIViewer.BODY_COLOR1[3]
        )
        self._body_color_1.set_on_value_changed(self._on_body_color_1)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Color"))
        grid.add_child(self._body_color_1)
        fitting_1.add_child(grid)

        self._stage_idx_1 = gui.Slider(gui.Slider.INT)
        self._stage_idx_1.set_limits(0, 0)
        self._stage_idx_1.set_on_value_changed(self._on_stage_idx_1)
        fitting_1.add_child(gui.Label("Stage"))
        fitting_1.add_child(self._stage_idx_1)  
        
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(fitting_1)
        
        #** fitting 2   
        fitting_2 = gui.CollapsableVert(
            "Fitting 2", 0.25 * em, gui.Margins(em, 0, 0, 0))
        
        self._fitting_combobox_2 = gui.Combobox()
        self._fitting_combobox_2.set_on_selection_changed(self._on_select_fitting_2)
        fitting_2.add_fixed(separation_height)
        fitting_2.add_child(self._fitting_combobox_2)
        
        self._body_color_2 = gui.ColorEdit()
        self._body_color_2.color_value = gui.Color(
            r=HSIViewer.BODY_COLOR2[0], g=HSIViewer.BODY_COLOR2[1],
            b=HSIViewer.BODY_COLOR2[2], a=HSIViewer.BODY_COLOR2[3]
        )
        self._body_color_2.set_on_value_changed(self._on_body_color_2)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Color"))
        grid.add_child(self._body_color_2)
        fitting_2.add_child(grid)  
        
        self._stage_idx_2 = gui.Slider(gui.Slider.INT)
        self._stage_idx_2.set_limits(0, 0)
        self._stage_idx_2.set_on_value_changed(self._on_stage_idx_2)
        fitting_2.add_child(gui.Label("Stage"))
        fitting_2.add_child(self._stage_idx_2)  
        
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(fitting_2)
        
        #* images panel
        self._images_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        
        self.images = {}
        for img_type in HSIViewer.IMAGE_TYPES:
            image = gui.CollapsableVert(
                img_type.capitalize() + " image", 0.25 * em, gui.Margins(em, 0, 0, 0))
            
            self.images[img_type] = gui.ImageWidget()
            self.images[img_type].update_image(self.default_img)
            image.add_fixed(separation_height)
            image.add_child(self.images[img_type])
        
            self._images_panel.add_fixed(separation_height)
            self._images_panel.add_child(image)
        
        w.add_child(self._scene)
        w.add_child(self._settings_panel)
        w.add_child(self._images_panel)
        
        w.set_on_layout(self._on_layout)
        
        #* menu
        if gui.Application.instance.menubar is None:
            file_menu = gui.Menu()
            file_menu.add_item("Export", HSIViewer.MENU_EXPORT)

            menu = gui.Menu()
            menu.add_menu("File", file_menu)
            gui.Application.instance.menubar = menu

        w.set_on_menu_item_activated(HSIViewer.MENU_EXPORT, self._on_menu_export)
      
    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        
        height_setting = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width, height_setting)

        height_images = min(
            r.height,
            self._images_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        self._images_panel.frame = gui.Rect(r.get_left(), r.y, width, height_images)


    def _on_select_dataset(self, dataset, idx):
        self._scene.scene.clear_geometry()
        for img_type in HSIViewer.IMAGE_TYPES:
            self.images[img_type].update_image(self.default_img)
        
        if dataset == 'PROXQ':
            self.dataio = data_io.PROXIO()
            recordings = [''] + constants.proxq_recordings
            fittings = [''] + constants.proxq_fittings
        elif dataset == 'PROX':
            self.dataio = data_io.PROXIO()
            recordings = [''] + constants.prox_recordings
            fittings = [''] + constants.prox_fittings
        else:
            self.dataio = None
            recordings = []
            fittings = []
        self.dataset = dataset
        
        self._recording_combobox.clear_items()
        for recording in recordings:
            self._recording_combobox.add_item(recording)
        
        self._frame_idx.set_limits(0, 0)
        self._frame_idx.int_value = 0
        
        self._fitting_combobox_1.clear_items()
        for fitting in fittings:
            self._fitting_combobox_1.add_item(fitting)
        self._fitting_combobox_2.clear_items()
        for fitting in fittings:
            self._fitting_combobox_2.add_item(fitting)
        self.fittings = [''] * 2
         
    def _on_select_recording(self, recording, idx):
        self._scene.scene.clear_geometry()
        
        if recording == '':
            return
        self.recording = recording
        self.dataio.instantiate(recording)
        if 'PROX' in self.dataset:
            use_quan = True if self.dataset == 'PROXQ' else False
            use_fixed = True if use_quan else False
            self.frame_ids = sorted(utils.get_frame_ids(recording, use_quan, use_fixed))
        self._frame_idx.set_limits(0, len(self.frame_ids)-1)
        self._frame_idx.int_value = 0
        
        self.safe_add_geometry('scene', self.dataio.scene_o3d)
        
        #* set shader to avoid the warning
        self.safe_add_geometry('camera', self.dataio.camera_o3d)
        self._scene.setup_camera(
            self.dataio.camera_k, self.dataio.trans_w2c, 
            constants.W, constants.H, self._scene.scene.bounding_box)
        
        if len(self.frame_ids) > 0:
            self.update_all()

    def _on_frame_idx(self, idx):
        self.update_all()

    def _on_mouse_backward(self):
        if self._frame_idx.int_value > self._frame_idx.get_minimum_value:
            self._frame_idx.int_value -= 1
        self.update_all()

    def _on_mouse_forward(self):
        if self._frame_idx.int_value < self._frame_idx.get_maximum_value:
            self._frame_idx.int_value += 1
        self.update_all()
    
    def _on_mouse_save(self):
        usage = ''
        
        frame_id = self.frame_ids[self._frame_idx.int_value]
        fit_idx = 0
        fit_name = self.fittings[fit_idx]
        stage_idx = str(int(self.stages[fit_idx]))
            
        #* make dir
        dir_name = self.recording + '-' + frame_id    
        save_dir = osp.join(constants.prox_mesh_root, usage, dir_name)  
        os.makedirs(save_dir, exist_ok=True)
        
        #* body
        body_mesh = trimesh.Trimesh(vertices=self.body_o3ds[fit_idx].vertices, 
                                    faces=self.body_o3ds[fit_idx].triangles)
        mesh_path = osp.join(save_dir, fit_name + '-' + stage_idx + '.ply')
        body_mesh.export(mesh_path)
            
        #* depth points
        depth_path = osp.join(save_dir, 'depth.ply')
        if not osp.exists(depth_path):
            depth_pc = trimesh.PointCloud(vertices=self.depth_o3d.points)
            depth_pc.export(depth_path)
            
        #* images
        depth_data = self.dataio.get_depth(frame_id, do_sample=False)
        color_path = osp.join(save_dir, 'color.jpg')
        os.system('cp ' + depth_data['color_path'] + ' ' + color_path)
        depth_path = osp.join(save_dir, 'depth.png')
        os.system('cp ' + depth_data['depth_path'] + ' ' + depth_path)
    
        #* rgb camera
        other_path = osp.join(save_dir, 'other.json')
        if not osp.exists(other_path):
            other_data = {}
            other_data['camera_pos'] = self.dataio.camera_pos.tolist()
            other_data['camera_int'] = self.dataio.camera_k.tolist()
            other_data['camera_ext'] = self.dataio.trans_w2c.tolist()
            other_json = json.dumps(other_data)
            with open(other_path, 'w') as f:
                f.write(other_json)
        
        #* current camera
        camera_idx = len(glob(save_dir + '/camera-*.json'))
        camera_path = osp.join(save_dir, 'camera-' + str(camera_idx) + '.json')
        camera_data = {}
        cur_model_mat = np.array(self._scene.scene.camera.get_model_matrix())
        cur_ext_mat = utils.model_to_ext(cur_model_mat)
        cur_fov = self._scene.scene.camera.get_field_of_view()
        camera_data['cur_model'] = cur_model_mat.tolist()
        camera_data['cur_ext'] = cur_ext_mat.tolist()
        camera_data['cur_fov'] = cur_fov
        camera_json = json.dumps(camera_data)
        with open(camera_path, 'w') as f:
            f.write(camera_json)
        
        #* window
        save_path = osp.join(save_dir, fit_name + '-' + stage_idx + '-' + str(camera_idx) + '.png')
        frame = self._scene.frame
        self.export_image(save_path, frame.width, frame.height)
        
    def _on_bg_color(self, color):
        color = [
            color.red, color.green, color.blue, color.alpha
        ]
        self._scene.scene.set_background(color)
   
    def _on_depth_color(self, color):
        self.materials['depth'].base_color = [
            color.red, color.green, color.blue, color.alpha
        ]
        self.safe_modify_geometry('depth')
    
    
    def _on_select_fitting_1(self, fitting, idx):
        self.fittings[0] = fitting
        self.reset_stage(0)
        self.update_body(1)
    
    def _on_body_color_1(self, color):
        self.materials['body1'].base_color = [
            color.red, color.green, color.blue, color.alpha
        ]
        self.safe_modify_geometry('body1')

    def _on_stage_idx_1(self, idx):
        self.stages[0] = idx
        self.update_body(1)

    def _on_select_fitting_2(self, fitting, idx):
        self.fittings[1] = fitting
        self.reset_stage(1)
        self.update_body(2)
    
    def _on_body_color_2(self, color):
        self.materials['body2'].base_color = [
            color.red, color.green, color.blue, color.alpha
        ]
        self.safe_modify_geometry('body2')

    def _on_stage_idx_2(self, idx):
        self.stages[1] = idx
        self.update_body(2)


    def update_all(self):
        self.update_depth()
        self.reset_stage(0)
        self.reset_stage(1)
        self.update_body(1)
        self.update_body(2)

    def update_depth(self):
        self.safe_remove_geometry('depth')
        
        frame_id = self.frame_ids[self._frame_idx.int_value]
        print(self.recording, frame_id)
        
        depth_data = self.dataio.get_depth(frame_id, do_sample=False)
        if depth_data['points'] is not None:
            self.depth_o3d.points = o3d.utility.Vector3dVector(depth_data['points'])
            self.safe_add_geometry('depth', self.depth_o3d)

        for img_type in HSIViewer.IMAGE_TYPES:
            img = depth_data[img_type]
            if img is None:
                continue
            #* bgr (cv2 default) -> rgb
            if 'color' in img_type:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img_type == 'depth':
                img *= 16.0
            img_o3d = o3d.geometry.Image(img.astype(np.uint8))
            self.images[img_type].update_image(img_o3d)

    def update_body(self, idx):
        body_name = 'body' + str(idx)
        idx -= 1
        fitting_name = self.fittings[idx]
        
        self.safe_remove_geometry(body_name)
        
        frame_id = self.frame_ids[self._frame_idx.int_value]
        stage_idx = self.stages[idx]
        body_data = self.dataio.get_body(frame_id, fitting_name, stage_idx)
        if body_data is not None:
            self.body_o3ds[idx].vertices = o3d.utility.Vector3dVector(body_data['vertices'])
            self.body_o3ds[idx].triangles = o3d.utility.Vector3iVector(self.dataio.model.faces)
            self.body_o3ds[idx].compute_vertex_normals()
            self.safe_add_geometry(body_name, self.body_o3ds[idx])
    
    
    def safe_add_geometry(self, name, o3d_object):
        self.safe_remove_geometry(name)
        try:
            self._scene.scene.add_geometry(name, o3d_object, self.materials[name])
        except:
            print('Error adding geometry, skip.')
             
    def safe_remove_geometry(self, name):
        if self._scene.scene.has_geometry(name):
            self._scene.scene.remove_geometry(name)
    
    def safe_modify_geometry(self, name):
        if self._scene.scene.has_geometry(name):
            self._scene.scene.modify_geometry_material(name, self.materials[name])
    
    def reset_stage(self, idx):
        if 'ours' in self.fittings[idx]:
            if idx == 0:
                self._stage_idx_1.set_limits(0, 7)
                self._stage_idx_1.int_value = 0
            elif idx == 1:
                self._stage_idx_2.set_limits(0, 7)
                self._stage_idx_2.int_value = 0
        else:
            if idx == 0:
                self._stage_idx_1.set_limits(0, 0)
                self._stage_idx_1.int_value = 0
            elif idx == 1:
                self._stage_idx_2.set_limits(0, 0)
                self._stage_idx_2.int_value = 0
        self.stages[idx] = 0
    
    
    def _on_menu_export(self):
        pass


    def export_image(self, path, width, height):
        def on_image(image):
            img = image

            quality = 9
            if path.endswith(".jpg"):
                quality = 100
            o3d.io.write_image(path, img, quality)

        self._scene.scene.scene.render_to_image(on_image)

    
def main():
    gui.Application.instance.initialize()

    w = HSIViewer(constants.W, constants.H)

    gui.Application.instance.run()


if __name__ == "__main__":
    main()