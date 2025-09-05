#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from pathlib import Path

import jinja2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--total",
        type=int,
        default=1,
        help="Number of runners",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of the current runner",
    )
    return parser.parse_args()


@dataclass
class PiperModel:
    # For en_GB-semaine-medium
    name: str  # semaine
    kind: str  # e.g. medium
    sr: int  # sample rate
    ns: int  # number of speakers
    checkpoint: str
    lang: str = ""  # e.g., en_GB
    cmd: str = ""
    model_name: str = ""
    text: str = ""
    index: int = 0
    url: str = ""


def get_zh_models():
    zh_cn = [
        PiperModel(
            name="huayan",
            kind="medium",
            sr=22050,
            ns=1,
            checkpoint="epoch=3269-step=2460540.ckpt",
        ),
    ]
    for m in zh_cn:
        m.lang = "zh_CN"

    ans = zh_cn

    for m in ans:
        m.text = "某某银行的副行长和一些行政领导表示，他们去过长江和长白山; 当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，思念如涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，感受着生命的奇迹与温柔."
        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.checkpoint}
            wget -qq https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/config.json
            wget -qq https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


def get_en_models():
    en_gb = [
        PiperModel(
            name="alan",
            kind="medium",
            sr=22050,
            ns=1,
            checkpoint="epoch=6339-step=1647790.ckpt",
        ),
        PiperModel(
            name="alba",
            kind="medium",
            sr=22050,
            ns=1,
            checkpoint="epoch=4179-step=2101090.ckpt",
        ),
        PiperModel(
            name="aru",
            kind="medium",
            sr=22050,
            ns=12,
            checkpoint="epoch=3479-step=939600.ckpt",
        ),
        PiperModel(
            name="cori",
            kind="high",
            sr=22050,
            ns=1,
            checkpoint="cori-high-500.ckpt",
        ),
        PiperModel(
            name="cori",
            kind="medium",
            sr=22050,
            ns=1,
            checkpoint="cori-med-640.ckpt",
            cmd="""
                   wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/cori/medium/en_GB-cori-medium.onnx.json
                   mv -v en_GB-cori-medium.onnx.json config.json

                   wget -qq https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_GB/cori/medium/cori-med-640.ckpt
                   """,
        ),
        PiperModel(
            name="jenny_dioco",
            kind="medium",
            sr=22050,
            ns=1,
            checkpoint="epoch=2748-step=1729300.ckpt",
        ),
        PiperModel(
            name="northern_english_male",
            kind="medium",
            sr=22050,
            ns=1,
            checkpoint="epoch=9029-step=2261720.ckpt",
        ),
        PiperModel(
            name="semaine",
            kind="medium",
            sr=22050,
            ns=4,
            checkpoint="epoch=1849-step=214600.ckpt",
        ),
        PiperModel(
            name="vctk",
            kind="medium",
            sr=22050,
            ns=109,
            checkpoint="epoch=545-step=1511328.ckpt",
        ),
    ]
    en_us = [
        PiperModel(
            name="amy",
            kind="medium",
            sr=22050,
            ns=1,
            checkpoint="epoch=6679-step=1554200.ckpt",
        ),
        PiperModel(
            name="arctic",
            kind="medium",
            sr=22050,
            ns=18,
            checkpoint="epoch=663-step=646736.ckpt",
        ),
        PiperModel(
            name="bryce", kind="medium", sr=22050, ns=1, checkpoint="bryce-3499.ckpt"
        ),
        PiperModel(
            name="hfc_female",
            kind="medium",
            sr=22050,
            ns=1,
            checkpoint="epoch=2868-step=1575188.ckpt",
        ),
        PiperModel(
            name="hfc_male",
            kind="medium",
            sr=22050,
            ns=1,
            checkpoint="epoch=2785-step=2128064.ckpt",
        ),
        PiperModel(
            name="joe",
            kind="medium",
            sr=22050,
            ns=1,
            checkpoint="epoch=7889-step=1221224.ckpt",
        ),
        PiperModel(
            name="john",
            kind="medium",
            sr=22050,
            ns=1,
            checkpoint="john-2599.ckpt",
        ),
        PiperModel(
            name="kristin",
            kind="medium",
            sr=22050,
            ns=1,
            checkpoint="kristin-2000.ckpt",
        ),
        PiperModel(
            name="kusal",
            kind="medium",
            sr=22050,
            ns=1,
            checkpoint="epoch=2652-step=1953828.ckpt",
        ),
        PiperModel(
            name="l2arctic",
            kind="medium",
            sr=22050,
            ns=24,
            checkpoint="epoch=536-step=902160.ckpt",
        ),
        PiperModel(
            name="lessac",
            kind="high",
            sr=22050,
            ns=1,
            checkpoint="epoch=2218-step=838782.ckpt",
        ),
        PiperModel(
            name="lessac",
            kind="low",
            sr=16000,
            ns=1,
            checkpoint="epoch=2307-step=558536.ckpt",
        ),
        PiperModel(
            name="lessac",
            kind="medium",
            sr=22050,
            ns=1,
            checkpoint="epoch=2164-step=1355540.ckpt",
        ),
        PiperModel(
            name="libritts_r",
            kind="medium",
            sr=22050,
            ns=904,
            checkpoint="epoch=404-step=1887300.ckpt",
        ),
        PiperModel(
            name="ljspeech",
            kind="high",
            sr=22050,
            ns=1,
            checkpoint="ljspeech-2000.ckpt",
        ),
        PiperModel(
            name="ljspeech",
            kind="medium",
            sr=22050,
            ns=1,
            checkpoint="lj-med_1000.ckpt",
        ),
        PiperModel(
            name="ryan",
            kind="medium",
            sr=22050,
            ns=1,
            checkpoint="epoch=4641-step=3104302.ckpt",
        ),
        PiperModel(
            name="sam",
            kind="medium",
            sr=22050,
            ns=1,
            checkpoint="epoch=4688-step=106008.ckpt",
        ),
    ]

    en_gb += [
        PiperModel(
            name="miro",
            kind="high",
            sr=22050,
            ns=1,
            checkpoint="epoch=9772-step=1494014.ckpt",
            cmd="""
                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_en-GB_miro/resolve/main/README.md

                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_en-GB_miro/resolve/main/miro_en-GB.onnx.json
                   mv -v miro_en-GB.onnx.json config.json

                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_en-GB_miro/resolve/main/epoch%3D9772-step%3D1494014.ckpt

                   echo "\n\nSee https://huggingface.co/OpenVoiceOS/pipertts_en-GB_miro" >> README.md
                   echo "and https://github.com/OHF-Voice/piper1-gpl/discussions/27" >> README.md
                   echo "\n\n# License\n\n" >> README.md

                   echo "See also https://github.com/k2-fsa/sherpa-onnx/pull/2480\n\n" >> README.md
                   echo "This model is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).\n" >> README.md

                   echo "- ✅ Always free for regular (non-commercial) users  \n" >> README.md
                   echo "- ❌ Commercial use is not allowed at this time  \n" >> README.md
                   echo "- 🔄 The author may relax the restrictions in the future (e.g., allow commercial use), but will not make them stricter  \n\n" >> README.md
                   echo "**Important:** You must include this license when redistributing the model or any derivatives.\n" >> README.md
                   """,
            url="https://huggingface.co/OpenVoiceOS/pipertts_en-GB_miro",
        ),
        PiperModel(
            name="dii",
            kind="high",
            sr=22050,
            ns=1,
            checkpoint="epoch=4610-step=1999288.ckpt",
            cmd="""
                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_en-GB_dii/resolve/main/README.md

                   echo "\n\nSee https://huggingface.co/OpenVoiceOS/pipertts_en-GB_dii" >> README.md
                   echo "and https://github.com/OHF-Voice/piper1-gpl/discussions/27" >> README.md
                   echo "\n\n# License\n\n" >> README.md

                   echo "See also https://github.com/k2-fsa/sherpa-onnx/pull/2480\n\n" >> README.md
                   echo "This model is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).\n" >> README.md

                   echo "- ✅ Always free for regular (non-commercial) users  \n" >> README.md
                   echo "- ❌ Commercial use is not allowed at this time  \n" >> README.md
                   echo "- 🔄 The author may relax the restrictions in the future (e.g., allow commercial use), but will not make them stricter  \n\n" >> README.md
                   echo "**Important:** You must include this license when redistributing the model or any derivatives.\n" >> README.md

                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_en-GB_dii/resolve/main/dii_en-GB.onnx.json
                   mv -v dii_en-GB.onnx.json config.json

                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_en-GB_dii/resolve/main/epoch%3D4610-step%3D1999288.ckpt
                   """,
            url="https://huggingface.co/OpenVoiceOS/pipertts_en-GB_dii",
        ),
    ]

    en_us.extend(
        [
            PiperModel(
                name="glados",
                kind="high",
                sr=22050,
                ns=1,
                checkpoint="last.ckpt",
                cmd="""
                   wget -qq https://huggingface.co/Shashashasha/glados-vits/resolve/main/last.ckpt
                   wget -qq https://huggingface.co/Shashashasha/glados-vits/resolve/main/config.json
                   """,
                url="https://huggingface.co/Shashashasha/glados-vits",
            ),
        ]
    )

    en_us += [
        PiperModel(
            name="miro",
            kind="high",
            sr=22050,
            ns=1,
            checkpoint="epoch=10638-step=1613522.ckpt",
            cmd="""
                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_en-US_miro/resolve/main/README.md

                   echo "\n\nSee https://huggingface.co/OpenVoiceOS/pipertts_en-US_miro" >> README.md
                   echo "and https://github.com/OHF-Voice/piper1-gpl/discussions/27" >> README.md
                   echo "\n\n# License\n\n" >> README.md

                   echo "See also https://github.com/k2-fsa/sherpa-onnx/pull/2480\n\n" >> README.md
                   echo "This model is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).\n" >> README.md

                   echo "- ✅ Always free for regular (non-commercial) users  \n" >> README.md
                   echo "- ❌ Commercial use is not allowed at this time  \n" >> README.md
                   echo "- 🔄 The author may relax the restrictions in the future (e.g., allow commercial use), but will not make them stricter  \n\n" >> README.md
                   echo "**Important:** You must include this license when redistributing the model or any derivatives.\n" >> README.md

                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_en-US_miro/resolve/main/epoch%3D10638-step%3D1613522.ckpt

                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_en-US_miro/resolve/main/miro_en-US.onnx.json
                   mv -v miro_en-US.onnx.json config.json
                   """,
            url="https://huggingface.co/OpenVoiceOS/pipertts_en-US_miro",
        ),
    ]

    for m in en_gb:
        m.lang = "en_GB"

    for m in en_us:
        m.lang = "en_US"

    ans = en_gb + en_us

    for m in ans:
        m.text = "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.checkpoint}
            wget -qq https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/config.json
            wget -qq https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


def get_all_models():
    ans = []
    ans += get_en_models()
    ans += get_zh_models()

    return ans


def main():
    args = get_args()
    index = args.index
    total = args.total
    assert 0 <= index < total, (index, total)

    all_model_list = get_all_models()

    num_models = len(all_model_list)
    num_per_runner = num_models // total
    if num_per_runner <= 0:
        raise ValueError(f"num_models: {num_models}, num_runners: {total}")

    start = index * num_per_runner
    end = start + num_per_runner

    remaining = num_models - args.total * num_per_runner

    print(f"{index}/{total}: {start}-{end}/{num_models}")

    d = dict()
    d["model_list"] = all_model_list[start:end]

    if index < remaining:
        s = args.total * num_per_runner + index
        d["model_list"].append(all_model_list[s])
        #  print(f"{s}/{num_models}")

    filename_list = [
        "./export.sh",
    ]
    for filename in filename_list:
        environment = jinja2.Environment()
        if not Path(f"{filename}.in").is_file():
            print(f"skip {filename}")
            continue

        with open(f"{filename}.in") as f:
            s = f.read()
        template = environment.from_string(s)

        s = template.render(**d)
        with open(filename, "w") as f:
            print(s, file=f)

    print(f"There are {len(all_model_list)} models")
    for m in all_model_list:
        print(m.index, m.name, m.kind)

    if Path("hf").is_dir():
        with open("./generate_samples.py.in") as f:
            s = f.read()
        template = environment.from_string(s)
        for m in all_model_list:
            model_dir = f"vits-piper-{m.lang}-{m.name}-{m.kind}"
            d = {
                "model": f"{model_dir}/{m.model_name}",
                "data_dir": f"{model_dir}/espeak-ng-data",
                "tokens": f"{model_dir}/tokens.txt",
                "text": m.text,
            }
            for i in range(m.ns):
                s = template.render(
                    **d,
                    sid=i,
                    output_filename=f"hf/piper/mp3/{m.lang}/{model_dir}/{i}.mp3",
                )

                with open(f"generate_samples-{model_dir}-{i}.py", "w") as f:
                    print(s, file=f)


if __name__ == "__main__":
    main()
