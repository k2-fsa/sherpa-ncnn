main.jar: Main.kt SherpaNcnn.kt AssetManager.kt WaveReader.kt
	kotlinc-jvm -include-runtime -d main.jar Main.kt SherpaNcnn.kt AssetManager.kt WaveReader.kt

run: main.jar
	java -Djava.library.path=../../build/lib -jar main.jar

