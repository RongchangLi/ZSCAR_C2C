# [ECCV2024 oral] C2C: Component-to-Composition Learning for Zero-Shot Compositional Action Recognition
### [Project Page](https://github.com/RongchangLi/ZSCAR_C2C) | [Paper](https://arxiv.org/abs/2407.06113)
<br/>

> C2C: Component-to-Composition Learning for Zero-Shot Compositional Action Recognition

> [Rongchang Li](https://rongchangli.github.io/), Zhenhua Feng, Tianyang Xu, Linze Li, Xiaojun Wu†, Muhammad Awais, Sara Atito, Josef Kittler           
> ECCV, 2024
                                                                 
[//]: # (&#40;For displaying sample GIFs&#41;)
<div align="center">
  <table style="border-collapse: collapse;">
    <tr>
      <td style="text-align: center; padding: 10px;">
        <img src="samples/open_door.gif" width="120" />
        <br />
        <i>
          <font color="black"><strong>Seen:</strong></font> 
          <font color="red">Open</font> 
          <font color="blue">a door</font>
        </i>
      </td>
      <td style="text-align: center; padding: 10px;">
        <img src="samples/close_book.gif" width="120" />
        <br />
        <i>
          <font color="black"><strong>Seen:</strong></font> 
          <font color="red">Close</font> 
          <font color="blue">a book</font>
        </i>
      </td>
      <td style="height: 120px; width: 1px; border-left: 2px dashed gray; text-align: center; padding: 10px;"></td>
      <td style="text-align: center; padding: 10px;">
        <img src="samples/close_door.gif" width="120" />
        <br />
        <i>
          <font color="black"><strong>Unseen:</strong></font> 
          <font color="red">Close</font> 
          <font color="blue">a door</font>
        </i>
      </td>
    </tr>
  </table>
  <div style="margin-top: 1px;">
    <strong>Zero-Shot Compositional Action Recognition (ZS-CAR)</strong>
  </div>
</div>


---
## 🛠️ Prepare Something-composition (Sth-com)
<p align="middle" style="margin-bottom: 0.5px;">
  <img src="samples/bend_spoon.gif" height="80" /> 
  <img src="samples/bend_book.gif" height="80" /> 
  <img src="samples/close_door.gif" height="80" /> 
  <img src="samples/close_book.gif" height="80" />
  <img src="samples/twist_obj.gif" height="80" /> 
</p>
<p align="middle" style="margin-bottom: 0.5px;margin-top: 0.5px;">
  <img src="samples/squeeze_bottle.gif" height="80" />
  <img src="samples/squeeze_pillow.gif" height="80" /> 
  <img src="samples/tear_card.gif" height="80" /> 
  <img src="samples/tear_leaf.gif" height="80" />
  <img src="samples/open_wallet.gif" height="80" />
</p>
<p align="center" style="margin-top: 0.5px;">
  <strong>Some samples in Something-composition</strong>
</p>

1. **Download Something-Something V2 (Sth-v2)**. Our proposed Something-composition (Sth-com) is based on [Sth-V2](https://developer.qualcomm.com/software/ai-datasets/something-something).
We refer to the official website to download the videos to the path _video_path_.
2. **Extract frames**. To accelerate the dataloader when training, we extract the frames for each video and save them in the _frame_path_. The command is:
      ```bash
      python tools/extract_frames.py --video_root video_path --frame_root frame_path
      ```
3. **Download Dataset annotations**. We provide our Sth-com annotation files in the [data_split](data_split/generalized) dir. The format is like:
    ```bash
      [
          {
          "id": "54463", # means the sample name
          "action": "opening a book", # means composition
          "verb": "Opening [something]", # means the verb component
          "object": "book" # means the object component
          },
          {
            ...
          },
          {
            ...
          },
      ]
    ```
    Please kindly download these files to _annotation_path_.
4. **Finally**, the dataset is built successfully. The structure looks like:
      >   * annotation_path
      >     * data_split
      >       * generalized
      >         * train_pairs.json
      >         * val_pairs.json
      >         * test_pairs.json
      >   * frame_path
      >     * 0
      >         * 000001.jpg
      >         * 000002.jpg
      >         * ......
      >     * 1
      >         * 000001.jpg
      >         * 000002.jpg
      >         * ......
      >     * ......

---
## 🚀 Train and test
🔔 Now take the dir **codes** as the project root. 

### Before running

1. Prepare the word embedding models. We recommend following [Compcos](https://github.com/ExplainableML/czsl) to download the word
   embedding models.
2. You should modify the paths :

   (For example, running C2C_vanilla, TSM-18 as the backbone.)
    1. _dataset_path_ in ./config/c2c_vanilla_tsm.yml
    2. _save_path_ in ./config/c2c_vanilla_tsm.yml
    3. The code line: _t=fasttext.load_model('YOUR_PATH/cc.en.300.bin')_ in models/vm_models/word_embedding.py

### Train

1. Train a model with the command:

  ```bash
 CUDA_VISIBLE_DEVICES=YOUR_GPU_INDEXEX python train.py --config config/c2c_vm/c2c_vanilla_tsm.yml
  ```

### Test

1. For the test, imagine you have trained your model and set the log dir as YOUR_LOG_PATH.

   Then, you can **test** it using:

  ```bash
 CUDA_VISIBLE_DEVICES=YOUR_GPU_INDEXEX python test_for_models.py --logpath YOUR_LOG_PATH
  ```

---
## 📝 TODO List
- [x] Add training codes for VM+word embedding paradigm.
- [x] Add training codes from VLM paradigm.
