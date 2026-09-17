# onnxruntime-android ships no consumer proguard rules (verified by
# inspecting the AAR); its native (C++) side calls back into these Java
# classes by name via JNI, so R8 must not rename or strip any of them.
# Deliberately package-wide since the exact JNI call sites aren't auditable
# from the managed side alone.
-keep class ai.onnxruntime.** { *; }
-keepclassmembers class ai.onnxruntime.** { *; }

# kotlinx.serialization and Coil both ship their own R8 consumer rules
# (META-INF/com.android.tools/r8/*.pro in their artifacts) — nothing to
# duplicate here for ing.emojify.model's @Serializable classes.
