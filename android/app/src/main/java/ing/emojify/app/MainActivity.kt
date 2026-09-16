package ing.emojify.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import ing.emojify.app.model.Meta
import ing.emojify.app.model.OnnxPredictor
import ing.emojify.app.ui.MainScreen
import kotlinx.serialization.json.Json

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val metaJson = assets.open("meta.json").bufferedReader().use { it.readText() }
        val meta = Json.decodeFromString(Meta.serializer(), metaJson)
        val predictor = OnnxPredictor(assets, meta)
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    MainScreen(meta = meta, predictor = predictor)
                }
            }
        }
    }
}
