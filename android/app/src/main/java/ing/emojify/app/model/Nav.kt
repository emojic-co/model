package ing.emojify.app.model

fun <T> cycle(list: List<T>, current: T?, dir: Int): T? {
    if (list.isEmpty()) return current
    val i = list.indexOf(current)
    if (i == -1) return if (dir > 0) list.first() else list.last()
    return list[(i + dir + list.size) % list.size]
}
