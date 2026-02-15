# Adat Előfeldolgozó (Interpoláció & Konvertálás)

Ez a script (`data_concat_with_interpolation.py`) a nyers, változó hosszúságú headset adatokat (`.csv`) dolgozza fel, és alakítja át egységes méretű NumPy tömbökké (`.npy`).

A program rekurzívan keresi a megadott CSV fájlokat a bemeneti mappában, hiánypótlást (interpolációt) végez rajtuk, majd egy rögzített időbeli hosszra (alapértelmezetten 2000 sor) méretezi át őket.

## Követelmények

A futtatáshoz szükséges Python környezet reprodukálható a `MOMENT/environment_data/` mappában található `.yml` fájl használatával:
```bash
conda env create -f environment.yml
```
<small>(Megjegyzés: Ha a fájl neve más, vagy máshol állsz a terminálban, igazítsd hozzá az útvonalat.)</small>

Amennyiben ez nem sikerülne akkor a következő csomagokkal kéne manuálisan telepíteni egy új környezetet:
* Python 3.8+ (ajánlott: 3.11)
* `pandas`
* `numpy`
* `scipy`
* `tqdm`

## Használat

A scriptet terminálból futtathatod paraméterek megadásával.

### Alapvető parancs
A legegyszerűbb futtatás, ha elfogadod az alapértelmezett útvonalakat (vagy ha a scriptben átírtad őket):
```bash
python data_concat_with_interpolation.py
```

### Egyedi könyvtárak megadása (Ajánlott)
Például ha más a célhossz vagy felül akarod írni a fájlokat:
```bash
python data_concat_with_interpolation.py --target_length 2048 --overwrite
```

## Argumentumok 
Az elérhető kapcsolók (flag-ek) részletes leírása:
| Argumentum | Leírás | Alapértelmezett Érték |
| :--- | :--- | :--- |
| `--input_dir` | A **bemeneti** főkönyvtár elérési útja. Ebben keresi a script a nyers CSV fájlokat. | `../zengo_recording` |
| `--output_dir` | A **kimeneti** főkönyvtár. Ide menti az elkészült .npy fájlokat, tükrözve az eredeti mappaszerkezetet. | `../zengo_preprocessed` |
| `--target_length` | A kimeneti idősorok kívánt hossza (sorok száma). Minden fájlt erre a méretre interpolál. | `2000` |
| `--target_filename` | A keresett fájlnév. A program csak az ilyen nevű fájlokat dolgozza fel. | `headset.csv` |
| `--overwrite` | Ha ezt a kapcsolót használod, a script **felülírja** a már létező .npy fájlokat. Enélkül átugorja őket. | *Kikapcsolva* |
| `--help`, `-h` | Megjeleníti a súgót és kilép. | - |

További használati információ eléréséhez használd:
```bash
python data_concat_with_interpolation.py --help
```